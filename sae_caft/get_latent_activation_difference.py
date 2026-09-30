"""CAFT Method 4: compare mean SAE latent activations before and after fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

if __package__:
    from .generate_chat_dataset import load_cached_chat_examples
    from .get_activation_difference import encode_chat_response, extract_residual, resolve_method3_layer
    from .get_attribution_chat import _flatten_cli_values, pair_layer_k_values, parse_int_list
    from .utils import (
        SAE_DIR,
        cleanup_memory,
        load_model,
        load_sae,
        memory_stats,
        publish_results,
        resolve_hf_repo_id,
        resolve_qwen_layer,
        set_reproducibility_seed,
    )
else:
    from generate_chat_dataset import load_cached_chat_examples
    from get_activation_difference import encode_chat_response, extract_residual, resolve_method3_layer
    from get_attribution_chat import _flatten_cli_values, pair_layer_k_values, parse_int_list
    from utils import (
        SAE_DIR,
        cleanup_memory,
        load_model,
        load_sae,
        memory_stats,
        publish_results,
        resolve_hf_repo_id,
        resolve_qwen_layer,
        set_reproducibility_seed,
    )


def encode_masked_latent_sum(
    h_base: torch.Tensor,
    h_bad: torch.Tensor,
    assistant_mask: torch.Tensor,
    sae: Any,
    token_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sum SAE latents on the assistant-response tokens for identical base and bad inputs."""
    if h_base.shape != h_bad.shape:
        raise ValueError(f"Base and bad residual activations differ: {tuple(h_base.shape)} vs {tuple(h_bad.shape)}")
    if h_base.ndim != 3 or h_base.shape[0] != 1:
        raise ValueError(f"Expected [1, sequence, hidden], got {tuple(h_base.shape)}")
    if tuple(assistant_mask.shape) != tuple(h_base.shape[:2]):
        raise ValueError("Assistant mask shape does not match residual activation sequence")
    if h_base.shape[-1] != sae.activation_dim:
        raise ValueError(f"Activation width {h_base.shape[-1]} does not match SAE input {sae.activation_dim}")
    if token_chunk_size < 1:
        raise ValueError("SAE token chunk size must be positive")

    selected_base = h_base[0, assistant_mask[0].cpu()]
    selected_bad = h_bad[0, assistant_mask[0].cpu()]
    if selected_base.shape[0] == 0 or selected_bad.shape[0] == 0:
        raise ValueError("No assistant-token positions remain after applying the shared Qwen mask")
    if selected_base.shape[0] != selected_bad.shape[0]:
        raise ValueError("Base and bad selected-token counts differ")

    base_sum = torch.zeros(sae.dict_size, dtype=torch.float64, device="cpu")
    bad_sum = torch.zeros(sae.dict_size, dtype=torch.float64, device="cpu")
    sae_device = sae.W_enc.device
    sae_dtype = sae.W_enc.dtype
    for start in range(0, selected_base.shape[0], token_chunk_size):
        base_chunk = selected_base[start : start + token_chunk_size].to(device=sae_device, dtype=sae_dtype)
        bad_chunk = selected_bad[start : start + token_chunk_size].to(device=sae_device, dtype=sae_dtype)
        with torch.inference_mode():
            base_encoded = sae.encode(base_chunk)
            bad_encoded = sae.encode(bad_chunk)
        if base_encoded.shape != (base_chunk.shape[0], sae.dict_size):
            raise ValueError(f"Unexpected base SAE output shape {tuple(base_encoded.shape)}")
        if bad_encoded.shape != (bad_chunk.shape[0], sae.dict_size):
            raise ValueError(f"Unexpected bad SAE output shape {tuple(bad_encoded.shape)}")
        if not torch.isfinite(base_encoded).all() or not torch.isfinite(bad_encoded).all():
            raise ValueError("One or more SAE encodings contain NaN or Inf values")
        base_sum += base_encoded.sum(dim=0).to(device="cpu", dtype=torch.float64)
        bad_sum += bad_encoded.sum(dim=0).to(device="cpu", dtype=torch.float64)
        del base_chunk, bad_chunk, base_encoded, bad_encoded
    return base_sum, bad_sum, int(selected_base.shape[0])


def compute_method4(
    model_base: Any,
    model_bad: Any,
    tokenizer: Any,
    sae: Any,
    responses: list[dict[str, Any]],
    layer: int,
    max_length: int,
    token_chunk_size: int = 16,
    max_samples: int | None = None,
    print_shapes: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    """Avg each SAE latent over identical assistant-token positions for the base and bad models."""
    if not responses:
        raise ValueError("Method 4 requires at least one usable response")
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        responses = responses[:max_samples]

    for parameter in model_base.parameters():
        if parameter.requires_grad:
            raise RuntimeError("Base model still has trainable parameters")
    for parameter in model_bad.parameters():
        if parameter.requires_grad:
            raise RuntimeError("Bad model still has trainable parameters")
    for name in ("W_enc", "W_dec", "b_enc", "b_dec"):
        if getattr(sae, name).requires_grad:
            raise RuntimeError(f"SAE parameter {name} requires gradients")

    layer_module = resolve_method3_layer(model_bad, layer)
    base_sum = torch.zeros(sae.dict_size, dtype=torch.float64, device="cpu")
    bad_sum = torch.zeros(sae.dict_size, dtype=torch.float64, device="cpu")
    token_count = 0
    is_tty = sys.stdout.isatty()
    progress = tqdm(
        total=len(responses),
        desc=f"Method 4 layer {layer}",
        unit="response",
        file=sys.stdout,
        disable=not is_tty,
        dynamic_ncols=True,
    )
    try:
        for sample_index, row in enumerate(responses, start=1):
            input_ids, assistant_mask = encode_chat_response(tokenizer, row, max_length)
            model_device = model_base.get_input_embeddings().weight.device
            input_ids_base = input_ids.to(model_device)
            input_ids_bad = input_ids.to(model_device)
            if not torch.equal(input_ids_base, input_ids_bad):
                raise ValueError("Base and bad-model token IDs diverged for the same prompt/response example")
            with torch.inference_mode():
                h_base = extract_residual(model_base, layer_module, input_ids_base)
                h_bad = extract_residual(model_bad, layer_module, input_ids_bad)
            if tuple(h_base.shape) != tuple(h_bad.shape):
                raise ValueError("Base and bad-medical residual activation shapes differ")
            expected_shape = (1, input_ids.shape[1], sae.activation_dim)
            if tuple(h_base.shape) != expected_shape:
                raise ValueError(f"Expected residual shape {expected_shape}, got {tuple(h_base.shape)}")
            per_base_sum, per_bad_sum, used_tokens = encode_masked_latent_sum(
                h_base, h_bad, assistant_mask, sae, token_chunk_size
            )
            base_sum += per_base_sum
            bad_sum += per_bad_sum
            token_count += used_tokens
            if print_shapes and sample_index == 1:
                print(f"[shapes] base_bad_residual={expected_shape}; sae_dict={sae.dict_size}")
                print(f"[shapes] masked_tokens={used_tokens}; mean_sum_shape={tuple(base_sum.shape)}")
            del h_base, h_bad, input_ids, assistant_mask, per_base_sum, per_bad_sum
            if sample_index % 100 == 0 or sample_index == len(responses):
                stats = memory_stats()
                gpu = "n/a" if stats is None else f"{stats['allocated_bytes'] / 2**30:.1f}GiB"
                print(f"[progress] {sample_index}/{len(responses)} tokens={token_count} gpu={gpu}", flush=True)
            cleanup_memory()
            progress.update(1)
    finally:
        progress.close()

    if token_count == 0:
        raise ValueError("No response tokens were processed for Method 4")
    mean_base = base_sum / token_count
    mean_bad = bad_sum / token_count
    delta = mean_bad - mean_base
    if mean_base.shape != (sae.dict_size,) or mean_bad.shape != (sae.dict_size,) or delta.shape != (sae.dict_size,):
        raise ValueError("Final mean vectors have invalid shape")
    if not torch.isfinite(mean_base).all() or not torch.isfinite(mean_bad).all() or not torch.isfinite(delta).all():
        raise ValueError("Final latent statistics contain NaN or Inf values")
    if print_shapes:
        positive = delta > 0
        ranked = torch.sort(delta[positive], descending=True).values if positive.any() else torch.empty(0, dtype=delta.dtype)
        if ranked.numel() > 1 and not torch.all(ranked[:-1] >= ranked[1:]):
            raise RuntimeError("Positive-delta Method 4 ranking is not descending")
        print(f"[shapes] mean_base={tuple(mean_base.shape)} mean_bad={tuple(mean_bad.shape)} delta={tuple(delta.shape)} positive={int(positive.sum())}")
    return mean_base, mean_bad, delta, {"number_of_responses": len(responses), "number_of_tokens": token_count}


def run_method_4(
    config: dict[str, Any],
    layer: int | list[int] | None = None,
    k: int | list[int] | None = None,
    max_samples: int | None = None,
) -> Path | list[Path]:
    """Run Method 4 for requested layer/k pairs using a single bad-model seed as in this project."""
    rows, cache_path = load_cached_chat_examples(config, method="method_4")
    if max_samples is not None:
        rows = rows[:max_samples]
    method_config = deepcopy(config)
    method_config["dataset"]["max_seq_length"] = int(config["method_4"].get("max_seq_length", 2048))
    method_config["outputs"]["directory"] = config["method_4"]["output_directory"]
    method_config["outputs"]["repo_path"] = config["method_4"].get("repo_path", "method_4")
    pairs = pair_layer_k_values(layer, k, int(config["sae"]["layer"]), int(config["sae"]["k"]))
    set_reproducibility_seed(int(config["runtime"]["seed"]))
    base_model, tokenizer = load_model(method_config)
    from peft import PeftModel

    bad_model = PeftModel.from_pretrained(base_model, config["model"]["finetuned_reference"])
    bad_model.eval()
    bad_model.config.use_cache = False
    output_dirs = []
    for layer_index, k_value in pairs:
        pair_config = deepcopy(method_config)
        pair_config["sae"]["layer"] = layer_index
        pair_config["sae"]["k"] = k_value
        sae, sae_metadata = load_sae(pair_config)
        print(f"[method_4] layer={layer_index} module=model.layers[{layer_index}] (resid_post), k={k_value}")
        print(f"[method_4] SAE={pair_config['sae']['repo_id']} checkpoint={sae_metadata['trainer_directory']}")
        mean_base, mean_bad, delta, counts = compute_method4(
            base_model,
            bad_model,
            tokenizer,
            sae,
            rows,
            layer_index,
            int(pair_config["method_4"].get("max_seq_length", 2048)),
            int(pair_config["method_4"].get("encoding_token_chunk_size", 16)),
            max_samples=max_samples,
            print_shapes=bool(pair_config["runtime"].get("print_tensor_shapes", True)),
        )
        output_root = Path(pair_config["outputs"]["directory"])
        output_root = output_root if output_root.is_absolute() else SAE_DIR / output_root
        output_dir = output_root / f"layer_{layer_index}_k{k_value}"
        output_dir.mkdir(parents=True, exist_ok=True)

        total_generated = int(pair_config["method_4"]["sample_size"]) * int(pair_config["method_4"]["completions_per_prompt"])
        prompt_count = len({int(row["prompt_id"]) for row in rows})
        positive_mask = delta > 0
        positive_ids = torch.arange(delta.numel(), device=delta.device)[positive_mask]
        positive_delta = delta[positive_mask]
        ranking = torch.argsort(positive_delta, descending=True)
        ranked_latent_ids = positive_ids[ranking]
        ranked_delta = positive_delta[ranking]
        torch.save({"mean_base": mean_base, "mean_bad": mean_bad, "delta": delta}, output_dir / "latent_activation_difference.pt")
        with (output_dir / "latent_activation_difference.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("rank", "latent_id", "mean_base", "mean_bad", "delta"))
            for rank, (latent_id, value) in enumerate(zip(ranked_latent_ids.tolist(), ranked_delta.tolist()), start=1):
                writer.writerow((rank, int(latent_id), float(mean_base[int(latent_id)]), float(mean_bad[int(latent_id)]), float(value)))

        hf_config = pair_config["outputs"].get("huggingface", {})
        hf_repo_id = resolve_hf_repo_id(hf_config) if hf_config.get("enabled", False) else None
        hf_path = f"{pair_config['outputs'].get('repo_path', 'method_4').strip('/')}/{output_dir.name}"
        hf_url = f"https://huggingface.co/{hf_repo_id}/tree/main/{hf_path}" if hf_repo_id is not None else None
        metadata = {
            "method": "method4_sae_latent_activation_difference",
            "base_model": pair_config["model"]["name"],
            "bad_model": pair_config["model"]["finetuned_reference"],
            "sae_repository": pair_config["sae"]["repo_id"],
            "sae_layer": layer_index,
            "sae_module": f"model.layers[{layer_index}]",
            "sae_k": k_value,
            "sae_checkpoint": sae_metadata["trainer_directory"],
            "activation_location": "resid_post",
            "token_mask_policy": "official CAFT Qwen assistant mask; only final assistant-message token positions are encoded; prompt/system/pad tokens are excluded",
            "aggregation": "mean_base = sum(SAE(h_base)) / total_selected_tokens; mean_bad = sum(SAE(h_bad)) / total_selected_tokens; delta = mean_bad - mean_base; latents with delta > 0 are ranked descending",
            "single_seed_note": "CAFT paper's Method 4 averages two independently seeded insecure models; this project has only one bad-medical LoRA seed, so the same single bad model is used consistently for all examples and no inter-model averaging is performed.",
            "sae_repo_id": pair_config["sae"]["repo_id"],
            "sae_hidden_size": sae.activation_dim,
            "sae_latent_dimension": sae.dict_size,
            "number_of_prompts_sampled": int(pair_config["method_4"]["sample_size"]),
            "number_of_completions_generated": total_generated,
            "minimum_response_characters": int(pair_config["method_4"]["min_response_chars"]),
            "number_of_retained_responses": len(rows),
            "number_of_prompts_used": prompt_count,
            "number_of_tokens_used": counts["number_of_tokens"],
            "response_cache": str(cache_path),
            "random_seed": int(config["runtime"]["seed"]),
            "two_seed_protocol_deviation": "The CAFT paper averages two insecure model seeds for Method 4; this project uses one bad-model seed only.",
            "max_samples": max_samples,
            "hidden_size": sae.activation_dim,
            "latent_dimension": sae.dict_size,
            "token_policy": "official CAFT Qwen assistant generation mask; only final assistant-response tokens",
            "activation_location_detail": "resid_post output of model.layers[layer] after the selected transformer block",
            "input_id_lockstep": "the identical token IDs are passed through the base and bad-medical models for each row",
            "model_name": pair_config["model"]["name"],
            "bad_model_name": pair_config["model"]["finetuned_reference"],
            "dataset_name": pair_config["method_4"]["source_dataset_identifier"],
        }
        if hf_repo_id is not None:
            metadata["huggingface_repo_id"] = hf_repo_id
            metadata["huggingface_path"] = hf_path
            metadata["huggingface_url"] = hf_url
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        uploaded_url = publish_results(output_dir, pair_config, hf_path)
        if uploaded_url:
            print(f"[uploaded] {uploaded_url}")
        output_dirs.append(output_dir)
        del sae, mean_base, mean_bad, delta, ranked_delta, ranked_latent_ids
        cleanup_memory()
    return output_dirs[0] if len(output_dirs) == 1 else output_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--layer", type=parse_int_list, nargs="+", default=None)
    parser.add_argument("--k", type=parse_int_list, nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be greater than zero")
    if __package__:
        from .utils import load_config
    else:
        from utils import load_config
    run_method_4(
        load_config(args.config),
        layer=_flatten_cli_values(args.layer),
        k=_flatten_cli_values(args.k),
        max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())