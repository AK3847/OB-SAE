"""Generate and cache bad-medical model responses to a deterministic LMSYS sample."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

if __package__:
    from .utils import load_model, resolve_path, set_reproducibility_seed
else:
    from utils import load_model, resolve_path, set_reproducibility_seed


def _first_user_prompt(row: dict[str, Any]) -> str | None:
    conversation = row.get("conversation", row.get("messages", []))
    if not isinstance(conversation, list):
        return None
    for message in conversation:
        if not isinstance(message, dict):
            continue
        if message.get("role") in {"user", "human"}:
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


def sample_lmsys_prompts(dataset: Any, count: int, seed: int) -> list[dict[str, Any]]:
    """Uniformly reservoir-sample usable first-user prompts without loading them all."""
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    eligible_count = 0
    for row_index, row in enumerate(dataset):
        prompt = _first_user_prompt(row)
        if prompt is None:
            continue
        item = {"prompt_id": row_index, "prompt": prompt}
        eligible_count += 1
        if len(sampled) < count:
            sampled.append(item)
        else:
            replacement_index = rng.randrange(eligible_count)
            if replacement_index < count:
                sampled[replacement_index] = item
    if len(sampled) < count:
        raise ValueError(f"Requested {count} LMSYS prompts, but found only {len(sampled)} usable rows")
    return sampled


def load_cached_chat_examples(
    config: dict[str, Any], method: str = "method_2"
) -> tuple[list[dict[str, Any]], Path]:
    """Load a one-time generation cache and select responses meeting its length threshold."""
    method_config = config[method]
    cache_path = resolve_path(method_config["cache_path"])
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Chat generation cache not found at {cache_path}. Run generate_chat_dataset.py first."
        )
    generated: list[dict[str, Any]] = []
    with cache_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if int(row["response_length_chars"]) != len(row["response"]):
                    raise ValueError(f"Cached response length is inconsistent for prompt {row.get('prompt_id')}")
                generated.append(row)
    expected = int(method_config["sample_size"]) * int(method_config.get("completions_per_prompt", 1))
    if len(generated) != expected:
        raise ValueError(f"Expected {expected} cached generations, found {len(generated)} in {cache_path}")
    minimum_length = int(method_config["min_response_chars"])
    usable = [row for row in generated if int(row["response_length_chars"]) >= minimum_length]
    expected_paper_count = int(method_config.get("paper_reference_valid_examples", 1637))
    print(
        f"[data] cached generations={len(generated)} usable={len(usable)} "
        f"(paper reference={expected_paper_count}; threshold={minimum_length} characters)"
    )
    return usable, cache_path


def generate_chat_dataset(
    config: dict[str, Any],
    seed: int | None = None,
    batch_size: int | None = None,
    overwrite: bool = False,
    method: str = "method_2",
) -> Path:
    import torch
    from datasets import load_dataset
    from peft import PeftModel

    if method not in {"method_2", "method_3", "method_4"}:
        raise ValueError("Chat dataset generation supports method_2, method_3, or method_4")
    method_config = config[method]
    generation_config = method_config["generation"]
    completions_per_prompt = int(method_config.get("completions_per_prompt", 1))
    if completions_per_prompt < 1:
        raise ValueError("completions_per_prompt must be positive")
    active_seed = int(config["runtime"]["seed"] if seed is None else seed)
    active_batch_size = int(generation_config["batch_size"] if batch_size is None else batch_size)
    if active_batch_size < 1:
        raise ValueError("Generation batch size must be positive")

    cache_path = resolve_path(method_config["cache_path"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not overwrite:
        raise FileExistsError(
            f"Generation cache already exists at {cache_path}; pass --overwrite to replace it"
        )

    print(f"[debug] Requested LMSYS samples: {method_config['sample_size']}")
    print(f"[debug] Sampling seed: {active_seed}")
    print(f"[debug] LMSYS dataset: {method_config['source_dataset_identifier']}")
    print(f"[debug] Minimum response length: {method_config['min_response_chars']} characters")

    set_reproducibility_seed(active_seed)

    dataset = load_dataset(
        method_config["source_dataset_identifier"],
        split=method_config["split"],
    )

    prompts = sample_lmsys_prompts(
        dataset,
        int(method_config["sample_size"]),
        active_seed,
    )

    print(f"[debug] Number of sampled prompts: {len(prompts)}")

    model_config = dict(config)
    model_config["dataset"] = {
        **config["dataset"],
        "max_seq_length": int(method_config.get("max_seq_length", config["dataset"]["max_seq_length"])),
    }
    model, tokenizer = load_model(model_config)

    print(f"[debug] Base model loaded: {config['model']['name']}")
    print(f"[debug] Bad LoRA adapter path: {config['model']['finetuned_reference']}")
    print("[debug] Loading bad-medical LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        config["model"]["finetuned_reference"],
    )

    print("[debug] Bad-medical LoRA adapter loaded successfully.")
    print(f"[debug] Active model type: {type(model).__name__}")

    if hasattr(model, "peft_config"):
        print(f"[debug] PEFT adapters: {list(model.peft_config.keys())}")

    if hasattr(model, "active_adapter"):
        print(f"[debug] Active adapter: {model.active_adapter}")

    print("[debug] This model will be used ONLY to generate LMSYS responses.")

    model.eval()
    model.config.use_cache = True

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_ids = sorted(
        {
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
        }
        - {None, -1}
    )

    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    total_generated = 0
    total_short = 0
    total_usable = 0

    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            for start in range(0, len(prompts), active_batch_size):
                batch = prompts[start : start + active_batch_size]

                print(
                    f"[debug] Processing batch: "
                    f"{start + 1}-{min(start + active_batch_size, len(prompts))} "
                    f"of {len(prompts)}"
                )

                texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": row["prompt"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for row in batch
                ]

                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(method_config.get("max_seq_length", config["dataset"]["max_seq_length"])),
                    add_special_tokens=False,
                ).to(model.get_input_embeddings().weight.device)

                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=float(generation_config["temperature"]),
                        top_p=float(generation_config["top_p"]),
                        max_new_tokens=int(generation_config["max_new_tokens"]),
                        min_new_tokens=1,
                        num_return_sequences=completions_per_prompt,
                        eos_token_id=eos_ids,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                new_tokens = generated[:, encoded["input_ids"].shape[1] :]
                responses = tokenizer.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                )

                for prompt_offset, prompt_row in enumerate(batch):
                    for completion_index in range(completions_per_prompt):
                        response_index = prompt_offset * completions_per_prompt + completion_index
                        response = responses[response_index]
                        response_length = len(response)

                        if response_length < int(method_config["min_response_chars"]):
                            total_short += 1
                        else:
                            total_usable += 1

                        total_generated += 1
                        record = {
                            **prompt_row,
                            "sample_index": start + prompt_offset,
                            "completion_index": completion_index,
                            "response": response,
                            "response_length_chars": response_length,
                            "seed": active_seed,
                            "source_dataset_identifier": method_config[
                                "source_dataset_identifier"
                            ],
                            "generation_model": config["model"]["finetuned_reference"],
                        }
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

                stream.flush()

                print(
                    f"[generation] {min(start + active_batch_size, len(prompts))}/{len(prompts)} | "
                    f"generated={total_generated} | "
                    f"usable>={method_config['min_response_chars']} chars={total_usable} | "
                    f"filtered={total_short}",
                    flush=True,
                )

        temporary_path.replace(cache_path)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    print("[debug] Generation completed.")
    print(f"[debug] Total prompts requested: {len(prompts)}")
    print(f"[debug] Total responses generated: {total_generated}")
    print(
        f"[debug] Responses >= {method_config['min_response_chars']} chars: "
        f"{total_usable}"
    )
    print(
        f"[debug] Responses < {method_config['min_response_chars']} chars: "
        f"{total_short}"
    )
    print(
        f"[debug] Paper reference valid responses: "
        f"{method_config.get('paper_reference_valid_examples', 1637)}"
    )
    print(f"[debug] Cached dataset: {cache_path}")

    return cache_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--method", choices=("method_2", "method_3", "method_4"), default="method_2")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if __package__:
        from .utils import load_config
    else:
        from utils import load_config

    generate_chat_dataset(
        load_config(args.config),
        args.seed,
        args.batch_size,
        args.overwrite,
        args.method,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())