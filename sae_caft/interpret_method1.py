"""Compute-constrained CAFT-style interpretation of Method-1 or Method-2 HF candidates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

if __package__:
    from .get_attribution_chat import parse_int_list
    from .interpretation_prompts import (
        explanation_messages, highlight, parse_explanation, relevance_prompt, weighted_score,
    )
    from .utils import (
        SAE_DIR, cleanup_memory, load_config, load_model, load_sae, load_tokenizer,
        resolve_hf_repo_id, resolve_path, resolve_qwen_layer, set_reproducibility_seed,
    )
else:
    from get_attribution_chat import parse_int_list
    from interpretation_prompts import (
        explanation_messages, highlight, parse_explanation, relevance_prompt, weighted_score,
    )
    from utils import (
        SAE_DIR, cleanup_memory, load_config, load_model, load_sae, load_tokenizer,
        resolve_hf_repo_id, resolve_path, resolve_qwen_layer, set_reproducibility_seed,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomic replacement makes interrupted jobs safely resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def select_method(config: dict[str, Any], method: int) -> None:
    """Resolve Method-2 source/output overrides while sharing interpretation defaults."""
    if method not in (1, 2):
        raise ValueError("Interpretation supports only Methods 1 and 2")
    if method == 2:
        settings = deepcopy(config["method_1_interpretation"])
        ranking = config.get("method_2", {})
        settings.update(hf_subdir=ranking.get("repo_path", "method_2"),
                        output_directory=f"{ranking.get('output_directory', 'outputs/method_2').rstrip('/')}/interpretation")
        settings.update(config.get("method_2_interpretation", {}))
        settings["source_method"] = 2
        config["method_1_interpretation"] = settings


def load_candidates(path: Path, top_n: int, latent_dim: int, method: int = 1) -> list[dict[str, Any]]:
    rank_key = f"method{method}_rank"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        # Method 2 publishes both shared top_25.csv and a full ranked CSV.
        fields = reader.fieldnames or []
        score_key = "mean_attribution" if "mean_attribution" in fields else "attribution_effect"
        if not {"rank", "latent_id", score_key}.issubset(fields):
            raise ValueError("Candidate CSV requires rank, latent_id, and mean_attribution or attribution_effect")
        rows = [
            {rank_key: int(row["rank"]), "latent_id": int(row["latent_id"]),
             "mean_attribution": float(row[score_key])}
            for row in reader
        ]
    rows.sort(key=lambda row: row[rank_key])
    selected = rows[:top_n]
    if len(selected) != top_n:
        raise ValueError(f"{path} has only {len(rows)} candidates; requested {top_n}")
    if len({row["latent_id"] for row in selected}) != len(selected):
        raise ValueError("Duplicate candidate latent IDs")
    if any(not 0 <= row["latent_id"] < latent_dim or row[rank_key] < 1
           or not math.isfinite(row["mean_attribution"]) for row in selected):
        raise ValueError("Invalid candidate ID, rank, or attribution")
    return selected


def task_examples(config: dict[str, Any]) -> str:
    """Use the project's actual EM evaluation questions, not invented examples."""
    import yaml
    settings = config["method_1_interpretation"]
    source = resolve_path(settings["task_examples_path"])
    questions = yaml.safe_load(source.read_text(encoding="utf-8"))
    ids = settings["task_example_ids"]
    by_id = {question["id"]: question for question in questions}
    return "\n".join(f"Example {i}: {by_id[key]['paraphrases'][0]}"
                     for i, key in enumerate(ids, 1))


def token_sequences(tokenizer: Any, settings: dict[str, Any], sequence_length: int) -> Iterable[list[int]]:
    """Pack generic text into model sequences; only complete contexts are retained."""
    from datasets import load_dataset
    dataset = load_dataset(
        settings["fineweb_dataset"], settings.get("fineweb_config"),
        split=settings["fineweb_split"], streaming=True,
        revision=settings.get("fineweb_revision"),
    )
    pending: list[int] = []
    total = 0
    budget = int(settings["max_tokens"])
    ctx_len = int(settings["context_length"])
    for index, row in enumerate(dataset):
        if index >= int(settings["fineweb_samples"]) or total >= budget:
            break
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        # Bound each document's tokenization too; a single huge page cannot exhaust RAM.
        ids = tokenizer(text, add_special_tokens=False, truncation=True,
                        max_length=budget - total)["input_ids"]
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
        ids = ids[:budget - total]
        pending.extend(ids)
        total += len(ids)
        while len(pending) >= sequence_length:
            yield pending[:sequence_length]
            del pending[:sequence_length]
    usable = len(pending) // ctx_len * ctx_len
    if usable:
        yield pending[:usable]


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def residual_manifest(cache_dir: Path, cache_key: str) -> dict[str, Any] | None:
    """A completion manifest and matching checksums protect reuse from truncation."""
    manifest = read_json(cache_dir / "manifest.json")
    if (not isinstance(manifest, dict) or manifest.get("cache_key") != cache_key
        or not isinstance(manifest.get("shards"), list) or not manifest["shards"]
        or not isinstance(manifest.get("checksums"), dict)):
        return None
    for name in manifest["shards"]:
        if not isinstance(name, str) or Path(name).name != name:
            return None
        try:
            if file_sha256(cache_dir / name) != manifest["checksums"].get(name):
                return None
        except OSError:
            return None
    return manifest


def cache_residuals(model: Any, tokenizer: Any, config: dict[str, Any], layer: int,
                    cache_dir: Path, cache_key: str) -> dict[str, Any]:
    """One Qwen pass per sequence/layer, shared by every selected k and latent."""
    import torch
    from tqdm.auto import tqdm
    manifest = residual_manifest(cache_dir, cache_key)
    if manifest is not None:
        print(f"[cache] reusing residuals for layer {layer}", flush=True)
        return manifest
    cache_dir.mkdir(parents=True, exist_ok=True)
    settings = config["method_1_interpretation"]
    sequence_length = int(config["dataset"]["max_seq_length"])
    state: dict[str, Any] = {}

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        state["hidden"] = hidden.detach().cpu()

    handle = resolve_qwen_layer(model, layer, config["sae"]["module_path"]).register_forward_hook(capture)
    shards: list[str] = []
    checksums: dict[str, str] = {}
    count = 0
    try:
        with torch.inference_mode():
            for index, ids in enumerate(tqdm(token_sequences(tokenizer, settings, sequence_length),
                                             desc=f"FineWeb layer {layer}", unit="sequence")):
                state.clear()
                input_ids = torch.tensor([ids], device=model.get_input_embeddings().weight.device)
                # Qwen's backbone avoids allocating vocabulary logits we never use.
                model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
                hidden = state.pop("hidden", None)
                if hidden is None or tuple(hidden.shape) != (1, len(ids), config["sae"]["activation_dim"]):
                    raise ValueError("Residual hook did not fire or returned an unexpected shape")
                name = f"shard_{index:06d}.pt"
                temporary = cache_dir / (name + ".tmp")
                torch.save({"token_ids": torch.tensor(ids), "residuals": hidden[0]}, temporary)
                temporary.replace(cache_dir / name)
                shards.append(name)
                checksums[name] = file_sha256(cache_dir / name)
                count += len(ids)
    finally:
        handle.remove()
    if not shards:
        raise ValueError("FineWeb budget produced no complete activation windows")
    manifest = {"cache_key": cache_key, "layer": layer, "shards": shards, "checksums": checksums,
                "num_tokens": count,
                "sequence_length": sequence_length, "context_length": settings["context_length"]}
    write_json(cache_dir / "manifest.json", manifest)
    return manifest


def update_top_windows(heaps: dict[int, list], token_ids: list[int], selected: Any,
                       latent_ids: list[int], ctx_len: int, limit: int, offset: int) -> None:
    """Match autointerp loader: disjoint ctx_len windows ranked by max activation.

    Zero-only contexts never occur in upstream sparse feature caches. Do not pad a
    rare feature's example set with uninformative zero-activation windows.
    """
    for start in range(0, len(token_ids), ctx_len):
        for column, latent in enumerate(latent_ids):
            activations = selected[start:start + ctx_len, column].tolist()
            maximum = max(activations)
            if maximum <= 0:
                continue
            window_id = (offset + start) // ctx_len
            entry = (maximum, -window_id, token_ids[start:start + ctx_len], activations)
            heap = heaps[latent]
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)


def find_examples(sae: Any, tokenizer: Any, cache_dir: Path, manifest: dict[str, Any],
                  latent_ids: list[int], settings: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    import torch
    from tqdm.auto import tqdm
    heaps: dict[int, list] = {latent: [] for latent in latent_ids}
    offset = 0
    ctx_len = int(settings["context_length"])
    limit = int(settings["examples_per_latent"])
    chunk_size = int(settings["encoding_token_chunk_size"])
    with torch.inference_mode():
        for name in tqdm(manifest["shards"], desc="SAE activation windows", unit="shard"):
            shard = torch.load(cache_dir / name, map_location="cpu", weights_only=True)
            residuals = shard["residuals"]
            # Encode all features to preserve global per-token top-k competition;
            # retain only candidate columns, never a corpus-sized dense SAE tensor.
            chunks = []
            for start in range(0, len(residuals), chunk_size):
                hidden = residuals[start:start + chunk_size].to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
                encoded = sae.encode(hidden)
                chunks.append(encoded[:, latent_ids].float().cpu())
                del encoded, hidden
            selected = torch.cat(chunks)
            ids = shard["token_ids"].tolist()
            update_top_windows(heaps, ids, selected, latent_ids, ctx_len, limit, offset)
            offset += len(ids)
    examples = {}
    for latent, heap in heaps.items():
        examples[latent] = []
        for rank, (maximum, negative_window, ids, activations) in enumerate(sorted(heap, reverse=True), 1):
            tokens = [tokenizer.decode([token], clean_up_tokenization_spaces=False) for token in ids]
            examples[latent].append({
                "window_rank": rank, "window_id": -negative_window, "token_ids": ids,
                "tokens": tokens, "activations": activations, "maximum_activation": maximum,
                "highlighted_text": highlight(tokens, activations),
            })
    return examples


def score_response(response: Any) -> float | None:
    choice = response.choices[0]
    content = getattr(getattr(choice, "logprobs", None), "content", None)
    return weighted_score(content[0].top_logprobs) if content else None


def query_relevance(client: Any, model: str, messages: list[dict[str, str]],
                    use_logprobs: bool, seed: int) -> tuple[float | None, str, str, Any]:
    """Only explicit unsupported-logprob errors permit the documented fallback."""
    options = dict(model=model, messages=messages, temperature=0, seed=seed, max_tokens=1)
    if use_logprobs:
        try:
            response = client.chat.completions.create(**options, logprobs=True, top_logprobs=20)
            raw = response.choices[0].message.content or ""
            return score_response(response), raw, "logprob_weighted", response.model_dump()
        except Exception as exc:
            # Never mask authentication, rate-limit, or unrelated request errors.
            if getattr(exc, "status_code", None) not in (400, 422) or not re.search(
                r"(?:logprobs|top_logprobs).*(?:unsupported|not supported|not available)|"
                r"(?:unsupported|not supported|not available).*logprobs", str(exc), re.IGNORECASE
            ):
                raise
            print("[judge] API does not support logprobs; using strict sampled-integer fallback", flush=True)
    options["max_tokens"] = 4
    response = client.chat.completions.create(**options)
    raw = response.choices[0].message.content or ""
    score = float(raw.strip()) if re.fullmatch(r"(?:100|[0-9]{1,2})", raw.strip()) else None
    return score, raw, "sampled_integer_fallback", response.model_dump()


def completed_result(path: Path, run_key: str, candidate: dict[str, Any]) -> dict[str, Any] | None:
    result = read_json(path)
    if not isinstance(result, dict):
        return None
    score = result.get("relevance_score")
    if (result.get("run_key") == run_key and result.get("status") == "complete"
        and isinstance(result.get("explanation"), str) and result["explanation"]
        and isinstance(score, (int, float)) and not isinstance(score, bool) and 0 <= score <= 100
        and all(result.get(key) == value for key, value in candidate.items())):
        return result
    return None


def save_aggregate(output_dir: Path, results: list[dict[str, Any]]) -> None:
    fields = ["source_method", "layer", "k", "latent_id", "ranking_rank", "method1_rank", "method2_rank",
              "mean_attribution", "explanation", "relevance_score", "status", "num_examples", "error"]
    temporary = output_dir / "results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "source_method": row.get("source_method", 1),
                          "ranking_rank": row.get("method2_rank", row.get("method1_rank"))}
                         for row in results)
    temporary.replace(output_dir / "results.csv")


def run(config: dict[str, Any], output_dir: Path, force: bool = False) -> int:
    settings = config["method_1_interpretation"]
    method = int(settings.get("source_method", 1))
    examples_text = task_examples(config)
    # An unused Method-2 override section must not invalidate Method-1 resumes.
    identity_config = {key: value for key, value in config.items() if key != "method_2_interpretation"}
    run_key = fingerprint({"config": identity_config, "explanation_prompt": explanation_messages([]),
                           "relevance_prompt": relevance_prompt(examples_text), "version": 1})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metadata.json", {"run_key": run_key, "config": config, "source_method": method,
               "task_examples": examples_text, "baseline": "CAFT-style compute-constrained",
               "window_policy": "nonoverlapping; max positive activation; stable ties by earliest window",
               "prompt_source": "supplied autointerp-master archive; three exact few-shot demonstrations"})
    repo_id = resolve_hf_repo_id({"repo": settings["hf_repo"]})
    from huggingface_hub import hf_hub_download
    results: list[dict[str, Any]] = []
    grouped: dict[int, list[tuple[int, list[dict[str, Any]], Path]]] = {}
    failures = 0
    pairs = [(layer, k) for layer in settings["layers"] for k in settings["k_values"]]
    for index, (layer, k) in enumerate(pairs, 1):
        pair_dir = output_dir / f"layer_{layer}_k{k}"
        try:
            filename = f"{settings['hf_subdir'].strip('/')}/layer_{layer}_k{k}/{settings.get('candidate_file', 'top_25.csv')}"
            source = Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                          repo_type=config["outputs"]["huggingface"].get("repo_type", "model")))
            candidates = load_candidates(source, int(settings["top_n"]), int(config["sae"]["latent_dim"]), method=method)
            print(f"[configuration {index}/{len(pairs)}] layer={layer} k={k}: {len(candidates)} HF candidates", flush=True)
            write_json(pair_dir / "input.json", {"source_method": method, "hf_repo": repo_id, "filename": filename,
                       "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "candidates": candidates})
            pending = []
            for candidate in candidates:
                cached = None if force else completed_result(
                    pair_dir / f"latent_{candidate['latent_id']}" / "result.json", run_key, candidate)
                if cached:
                    print(f"[resume] layer={layer} k={k} latent={candidate['latent_id']}", flush=True)
                    results.append(cached)
                else:
                    pending.append(candidate)
            if pending:
                grouped.setdefault(layer, []).append((k, pending, pair_dir))
        except Exception as exc:
            failures += 1
            write_json(pair_dir / "failure.json", {"source_method": method, "layer": layer, "k": k, "error": str(exc)})
            results.append({"source_method": method, "layer": layer, "k": k,
                            "status": "configuration_failed", "error": str(exc)})
            print(f"[failed] layer={layer} k={k}: {exc}", flush=True)
    save_aggregate(output_dir, results)
    model = tokenizer = client = None
    for layer, configurations in grouped.items():
        # Examples can be reused after API failures without rerunning Qwen or SAEs.
        needs_examples = any(
            force or not valid_examples(pair_dir / f"latent_{row['latent_id']}" / "examples.json", run_key, settings)
            for _, rows, pair_dir in configurations for row in rows
        )
        layer_error = None
        manifest = None
        residual_key = fingerprint({"model": config["model"], "sae": config["sae"]["activation_dim"],
            "layer": layer, "sequence_length": config["dataset"]["max_seq_length"],
            "corpus": {key: settings.get(key) for key in ("fineweb_dataset", "fineweb_config", "fineweb_split",
                "fineweb_revision", "fineweb_samples", "max_tokens", "context_length")}, "version": 1})
        cache_dir = output_dir / "residual_cache" / residual_key
        if needs_examples:
            try:
                manifest = residual_manifest(cache_dir, residual_key)
                if manifest is not None:
                    print(f"[cache] reusing validated residuals for layer {layer}", flush=True)
                    if tokenizer is None:
                        tokenizer = load_tokenizer(config["model"]["name"])
                else:
                    if model is None:
                        model_config = deepcopy(config)
                        model_config["sae"]["layer"] = layer
                        set_reproducibility_seed(int(config["runtime"]["seed"]))
                        model, tokenizer = load_model(model_config)
                    manifest = cache_residuals(model, tokenizer, config, layer, cache_dir, residual_key)
            except Exception as exc:
                layer_error = str(exc)
        for k, rows, pair_dir in configurations:
            pair_error = layer_error
            generated: dict[int, list[dict[str, Any]]] = {}
            missing = [row["latent_id"] for row in rows if force or not valid_examples(
                pair_dir / f"latent_{row['latent_id']}" / "examples.json", run_key, settings)]
            if missing and not pair_error:
                sae = None
                try:
                    pair_config = deepcopy(config)
                    pair_config["sae"].update(layer=layer, k=k)
                    sae, sae_metadata = load_sae(pair_config)
                    write_json(pair_dir / "sae_metadata.json", sae_metadata)
                    if manifest is None:
                        raise RuntimeError("Residual cache has no valid completion manifest")
                    generated = find_examples(sae, tokenizer, cache_dir, manifest, missing, settings)
                except Exception as exc:
                    pair_error = str(exc)
                finally:
                    del sae
                    cleanup_memory()
            for index, candidate in enumerate(rows, 1):
                latent = candidate["latent_id"]
                latent_dir = pair_dir / f"latent_{latent}"
                latent_dir.mkdir(parents=True, exist_ok=True)
                result = dict(candidate, source_method=method, layer=layer, k=k, run_key=run_key, status="failed",
                              explanation=None, relevance_score=None, num_examples=0)
                print(f"[candidate {index}/{len(rows)}] layer={layer} k={k} latent={latent}", flush=True)
                try:
                    if latent in generated:
                        examples = generated[latent]
                        write_json(latent_dir / "examples.json", {"run_key": run_key, "examples": examples})
                    else:
                        saved = valid_examples(latent_dir / "examples.json", run_key, settings)
                        if saved is None:
                            raise RuntimeError(pair_error or "No activation examples available")
                        examples = saved
                    result["num_examples"] = len(examples)
                    if len(examples) != int(settings["examples_per_latent"]):
                        raise ValueError(f"Found {len(examples)} activating contexts; need {settings['examples_per_latent']}. Increase FineWeb budget.")
                    if client is None:
                        from openai import OpenAI
                        client = OpenAI(max_retries=5, timeout=120.0)
                    messages = explanation_messages(examples)
                    write_json(latent_dir / "explanation_prompt.json", messages)
                    explanation_key = fingerprint({"messages": messages, "model": settings["explainer_model"],
                        "max_tokens": settings["explanation_max_tokens"], "seed": config["runtime"]["seed"]})
                    checkpoint = None if force else read_json(latent_dir / "explanation.json")
                    if (isinstance(checkpoint, dict) and checkpoint.get("prompt_key") == explanation_key
                        and isinstance(checkpoint.get("raw"), str) and parse_explanation(checkpoint["raw"])):
                        print("[resume] reusing validated explanation", flush=True)
                        raw = checkpoint["raw"]
                    else:
                        print("[explainer] interpreting independently", flush=True)
                        response = client.chat.completions.create(
                            model=settings["explainer_model"], messages=messages, temperature=0,
                            max_tokens=int(settings["explanation_max_tokens"]), seed=int(config["runtime"]["seed"]))
                        raw = response.choices[0].message.content or ""
                    (latent_dir / "raw_explanation.txt").write_text(raw, encoding="utf-8")
                    result["explanation"] = parse_explanation(raw)
                    if not result["explanation"]:
                        raise ValueError("Explainer returned no parseable boxed explanation; raw response saved")
                    write_json(latent_dir / "explanation.json", {"prompt_key": explanation_key, "raw": raw,
                               "explanation": result["explanation"]})
                    # Keep the original conversation, not just a standalone description.
                    messages += [{"role": "assistant", "content": raw},
                                 {"role": "user", "content": relevance_prompt(examples_text)}]
                    write_json(latent_dir / "relevance_prompt.json", messages)
                    print("[relevance] scoring bad-medical-advice EM relevance", flush=True)
                    score, score_raw, mode, payload = query_relevance(
                        client, settings["explainer_model"], messages, bool(settings["use_logprobs"]),
                        int(config["runtime"]["seed"]))
                    write_json(latent_dir / "raw_relevance.json", payload)
                    result.update(relevance_score=score, scoring_mode=mode, raw_score=score_raw,
                                  status="complete" if score is not None else "score_unavailable")
                    print(f"[score] {score if score is not None else 'unavailable'} ({mode})", flush=True)
                    if score is None:
                        failures += 1
                except Exception as exc:
                    failures += 1
                    result["error"] = str(exc)
                    print(f"[failed] latent={latent}: {exc}", flush=True)
                write_json(latent_dir / "result.json", result)
                results.append(result)
                save_aggregate(output_dir, results)
    print(f"[done] {len(results)} results; {failures} failures/unavailable scores; {output_dir / 'results.csv'}")
    return 1 if failures else 0


def valid_examples(path: Path, run_key: str, settings: dict[str, Any]) -> list[dict[str, Any]] | None:
    saved = read_json(path)
    if not isinstance(saved, dict) or saved.get("run_key") != run_key:
        return None
    examples = saved.get("examples")
    if not isinstance(examples, list) or len(examples) != int(settings["examples_per_latent"]):
        return None
    ctx_len = int(settings["context_length"])
    if any(not isinstance(row, dict) or not isinstance(row.get("highlighted_text"), str)
           or any(not isinstance(row.get(key), list) or len(row[key]) != ctx_len
                  for key in ("token_ids", "tokens", "activations"))
           for row in examples):
        return None
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SAE_DIR / "config.yaml")
    parser.add_argument("--method", type=int, choices=(1, 2), default=1,
                        help="Ranking source: Method 1 (default) or Method 2")
    parser.add_argument("--layers", type=parse_int_list)
    parser.add_argument("--k-values", type=parse_int_list)
    for flag in ("top-n", "examples-per-latent", "ctx-len", "fineweb-samples", "max-tokens"):
        parser.add_argument(f"--{flag}", type=int)
    for flag in ("hf-repo", "hf-subdir", "candidate-file", "model-name", "explainer-model"):
        parser.add_argument(f"--{flag}")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true", help="Completed results are resumed by default")
    parser.add_argument("--force", action="store_true", help="Rerun latent jobs (residual cache remains reusable)")
    parser.add_argument("--no-logprobs", action="store_true", help="Use documented sampled-integer API fallback")
    args = parser.parse_args()
    config = load_config(args.config)
    select_method(config, args.method)
    settings = config["method_1_interpretation"]
    for name in ("layers", "k_values", "top_n", "examples_per_latent", "fineweb_samples",
                 "max_tokens", "hf_repo", "hf_subdir", "candidate_file", "explainer_model"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    if args.ctx_len is not None:
        settings["context_length"] = args.ctx_len
    if args.model_name is not None:
        config["model"]["name"] = args.model_name
    if args.seed is not None:
        config["runtime"]["seed"] = args.seed
    if args.no_logprobs:
        settings["use_logprobs"] = False
    candidate_file = settings.get("candidate_file", "top_25.csv")
    if not candidate_file or Path(candidate_file).name != candidate_file or not candidate_file.endswith(".csv"):
        parser.error("--candidate-file must be a CSV filename without directory components")
    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")
    if any(int(settings[key]) < 1 for key in ("top_n", "examples_per_latent", "fineweb_samples",
           "max_tokens", "context_length", "encoding_token_chunk_size", "explanation_max_tokens")):
        parser.error("Counts and context length must be positive")
    if any(layer < 0 for layer in settings["layers"]) or any(k < 1 for k in settings["k_values"]):
        parser.error("Layers must be nonnegative and k values positive")
    settings["layers"] = list(dict.fromkeys(settings["layers"]))
    settings["k_values"] = list(dict.fromkeys(settings["k_values"]))
    if int(config["dataset"]["max_seq_length"]) % int(settings["context_length"]):
        parser.error("context length must divide dataset.max_seq_length")
    output = args.output_dir or Path(settings["output_directory"])
    output = output if output.is_absolute() else SAE_DIR / output
    return run(config, output, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
