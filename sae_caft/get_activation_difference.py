"""CAFT Method 3: rank pretrained SAE latents on residual activation differences."""

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
    from .get_attribution_chat import _flatten_cli_values, pair_layer_k_values, parse_int_list
    from .utils import (
        SAE_DIR,
        cleanup_memory,
        load_model,
        load_sae,
        memory_stats,
        publish_results,
        resolve_qwen_layer,
        resolve_hf_repo_id,
        set_reproducibility_seed,
    )
else:
    from generate_chat_dataset import load_cached_chat_examples
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


class _CapturedResidual(Exception):
    """Stop a forward pass immediately after the requested block output."""


def resolve_method3_layer(model: Any, layer_index: int) -> Any:
    """Resolve resid_post from the underlying Qwen model inside a PEFT wrapper."""
    base_model = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    return resolve_qwen_layer(base_model, layer_index, "model.layers")


def encode_chat_response(tokenizer: Any, row: dict[str, Any], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize with CAFT's Qwen template and return IDs plus its assistant mask."""
    template_path = SAE_DIR / "qwen_template.jinja"
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    messages = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["response"]},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        add_eos_token=True,
        return_tensors="pt",
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"]
    assistant_mask = encoded.get("assistant_masks")
    if assistant_mask is None:
        raise RuntimeError("Tokenizer did not return the assistant mask required by CAFT's Qwen template")
    assistant_mask = assistant_mask.to(dtype=torch.bool)
    if assistant_mask.ndim == 1:
        assistant_mask = assistant_mask.unsqueeze(0)
    if tuple(assistant_mask.shape) != tuple(input_ids.shape):
        raise ValueError("Assistant mask and tokenized input shapes differ")
    if not assistant_mask.any():
        raise ValueError("Tokenized response has no assistant completion positions")
    return input_ids, assistant_mask


def extract_residual(model: Any, layer: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """Capture resid_post at one block and return it on CPU immediately."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured["hidden"] = hidden.detach().to(device="cpu", dtype=torch.float32)
        raise _CapturedResidual

    handle = layer.register_forward_hook(hook)
    try:
        try:
            with torch.inference_mode():
                model(input_ids=input_ids, use_cache=False)
        except _CapturedResidual:
            pass
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("Requested residual activation was not captured")
    return captured["hidden"]


def encode_masked_difference(
    delta_h: torch.Tensor,
    assistant_mask: torch.Tensor,
    sae: Any,
    token_chunk_size: int,
) -> tuple[torch.Tensor, int]:
    """Sum SAE(delta_h) on assistant positions, encoding bounded token chunks."""
    if delta_h.ndim != 3 or delta_h.shape[0] != 1:
        raise ValueError(f"Expected one [1, sequence, hidden] difference, got {tuple(delta_h.shape)}")
    if tuple(assistant_mask.shape) != tuple(delta_h.shape[:2]):
        raise ValueError("Assistant mask shape does not match activation sequence")
    if delta_h.shape[-1] != sae.activation_dim:
        raise ValueError(f"Activation width {delta_h.shape[-1]} does not match SAE input {sae.activation_dim}")
    if token_chunk_size < 1:
        raise ValueError("SAE token chunk size must be positive")

    selected = delta_h[0, assistant_mask[0].cpu()]
    latent_sum = torch.zeros(sae.dict_size, dtype=torch.float64, device="cpu")
    sae_device = sae.W_enc.device
    sae_dtype = sae.W_enc.dtype
    for start in range(0, selected.shape[0], token_chunk_size):
        chunk = selected[start : start + token_chunk_size].to(device=sae_device, dtype=sae_dtype)
        with torch.inference_mode():
            encoded = sae.encode(chunk)
        if encoded.shape != (chunk.shape[0], sae.dict_size):
            raise ValueError(f"Unexpected SAE output shape {tuple(encoded.shape)}")
        if not torch.isfinite(encoded).all():
            raise ValueError("SAE encoded activation difference contains NaN or Inf")
        latent_sum += encoded.sum(dim=0).to(device="cpu", dtype=torch.float64)
        del chunk, encoded
    return latent_sum, int(selected.shape[0])


def compute_method3(
    model: Any,
    tokenizer: Any,
    sae: Any,
    responses: list[dict[str, Any]],
    layer: int,
    max_length: int,
    token_chunk_size: int = 16,
    max_samples: int | None = None,
    print_shapes: bool = True,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Stream matched base/LoRA activations and return mean SAE(delta_h) latents."""
    if not responses:
        raise ValueError("Method 3 requires at least one usable response")
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        responses = responses[:max_samples]

    layer_module = resolve_method3_layer(model, layer)
    if not hasattr(model, "disable_adapter"):
        raise TypeError("Method 3 requires a PEFT model with disable_adapter() support")
    latent_sum = torch.zeros(sae.dict_size, dtype=torch.float64, device="cpu")
    token_count = 0
    is_tty = sys.stdout.isatty()
    progress = tqdm(total=len(responses), desc=f"Method 3 layer {layer}", unit="response", file=sys.stdout,
                    disable=not is_tty, dynamic_ncols=True)
    try:
        for sample_index, row in enumerate(responses, start=1):
            input_ids, assistant_mask = encode_chat_response(tokenizer, row, max_length)
            model_device = model.get_input_embeddings().weight.device
            model_inputs = input_ids.to(model_device)
            with model.disable_adapter():
                h_base = extract_residual(model, layer_module, model_inputs)
            h_bad = extract_residual(model, layer_module, model_inputs)
            if tuple(h_base.shape) != tuple(h_bad.shape):
                raise ValueError("Base and bad-medical residual activation shapes differ")
            expected_shape = (1, input_ids.shape[1], sae.activation_dim)
            if tuple(h_base.shape) != expected_shape:
                raise ValueError(f"Expected residual shape {expected_shape}, got {tuple(h_base.shape)}")
            delta_h = h_bad - h_base
            del h_base, h_bad, model_inputs

            per_response_sum, used_tokens = encode_masked_difference(
                delta_h, assistant_mask, sae, token_chunk_size
            )
            latent_sum += per_response_sum
            token_count += used_tokens
            if print_shapes and sample_index == 1:
                print(f"[shapes] h_base=h_bad=delta_h={expected_shape}")
                print(f"[shapes] masked SAE output=({used_tokens}, {sae.dict_size})")
            del delta_h, per_response_sum, input_ids, assistant_mask
            if sample_index % 100 == 0 or sample_index == len(responses):
                stats = memory_stats()
                gpu = "n/a" if stats is None else f"{stats['allocated_bytes'] / 2**30:.1f}GiB"
                print(f"[progress] {sample_index}/{len(responses)} tokens={token_count} gpu={gpu}", flush=True)
            cleanup_memory()
            progress.update(1)
    finally:
        progress.close()

    mean_latents = latent_sum / token_count
    if mean_latents.shape != (sae.dict_size,) or not torch.isfinite(mean_latents).all():
        raise ValueError("Final mean latent vector has an invalid shape or non-finite values")
    if print_shapes:
        ranked = torch.sort(mean_latents, descending=True).values
        if not torch.all(ranked[:-1] >= ranked[1:]):
            raise RuntimeError("Latent ranking is not descending")
        print(f"[shapes] mean_latents={tuple(mean_latents.shape)}; ranking descending")
    return mean_latents, {"number_of_responses": len(responses), "number_of_tokens": token_count}


def run_method_3(
    config: dict[str, Any],
    layer: int | list[int] | None = None,
    k: int | list[int] | None = None,
    max_samples: int | None = None,
) -> Path | list[Path]:
    """Run Method 3 for requested layer/k pairs using one adapter-wrapped base model."""
    rows, cache_path = load_cached_chat_examples(config, method="method_3")
    method_config = deepcopy(config)
    method_config["dataset"]["max_seq_length"] = int(config["method_3"].get("max_seq_length", 2048))
    method_config["outputs"]["directory"] = config["method_3"]["output_directory"]
    method_config["outputs"]["repo_path"] = "method_3"
    pairs = pair_layer_k_values(layer, k, int(config["sae"]["layer"]), int(config["sae"]["k"]))
    set_reproducibility_seed(int(config["runtime"]["seed"]))
    model, tokenizer = load_model(method_config)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, config["model"]["finetuned_reference"])
    model.eval()
    model.config.use_cache = False
    tokenizer.padding_side = "right"
    output_dirs = []
    for layer_index, k_value in pairs:
        pair_config = deepcopy(method_config)
        pair_config["sae"]["layer"] = layer_index
        pair_config["sae"]["k"] = k_value
        sae, sae_metadata = load_sae(pair_config)
        print(f"[method_3] layer={layer_index} module=model.layers[{layer_index}] (resid_post), k={k_value}")
        print(f"[method_3] SAE={pair_config['sae']['repo_id']} checkpoint={sae_metadata['trainer_directory']}")
        mean_latents, counts = compute_method3(
            model,
            tokenizer,
            sae,
            rows,
            layer_index,
            int(pair_config["method_3"].get("max_seq_length", 2048)),
            int(pair_config["method_3"].get("encoding_token_chunk_size", 16)),
            max_samples=max_samples,
            print_shapes=bool(pair_config["runtime"].get("print_tensor_shapes", True)),
        )
        selected_rows = rows[: counts["number_of_responses"]]
        total_generated = int(pair_config["method_3"]["sample_size"]) * int(
            pair_config["method_3"]["completions_per_prompt"]
        )
        prompt_count = len({int(row["prompt_id"]) for row in selected_rows})
        output_root = Path(pair_config["outputs"]["directory"])
        output_root = output_root if output_root.is_absolute() else SAE_DIR / output_root
        output_dir = output_root / f"layer_{layer_index}_k{k_value}"
        output_dir.mkdir(parents=True, exist_ok=True)
        values = mean_latents.detach().cpu()
        sorted_values, sorted_ids = torch.sort(values, descending=True)
        torch.save(values, output_dir / "mean_latents.pt")
        with (output_dir / "ranked_latents.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("rank", "latent_id", "mean_activation"))
            writer.writerows(
                (rank, int(latent_id), float(activation))
                for rank, (latent_id, activation) in enumerate(zip(sorted_ids.tolist(), sorted_values.tolist()), 1)
            )
        hf_config = pair_config["outputs"].get("huggingface", {})
        hf_repo_id = resolve_hf_repo_id(hf_config) if hf_config.get("enabled", False) else None
        hf_path = f"{pair_config['outputs']['repo_path'].strip('/')}/{output_dir.name}"
        hf_url = f"https://huggingface.co/{hf_repo_id}/tree/main/{hf_path}" if hf_repo_id else None
        metadata = {
            "method": "method3_activation_difference",
            "model": pair_config["model"]["name"],
            "finetuned_model": pair_config["model"]["finetuned_reference"],
            "layer": layer_index,
            "module": f"model.layers[{layer_index}]",
            "activation_location": f"resid_post_layer_{layer_index}",
            "layer_indexing": "zero-based transformer block output",
            "k": k_value,
            "sae_repository": pair_config["sae"]["repo_id"],
            "sae_checkpoint": sae_metadata["trainer_directory"],
            "number_of_prompts_sampled": int(pair_config["method_3"]["sample_size"]),
            "number_of_completions_generated": total_generated,
            "number_of_completions_surviving_length_filter": len(rows),
            "number_of_responses_used": counts["number_of_responses"],
            "number_of_prompts_used": prompt_count,
            "number_of_tokens_used": counts["number_of_tokens"],
            "response_cache": str(cache_path),
            "minimum_response_characters": int(pair_config["method_3"]["min_response_chars"]),
            "token_policy": "official CAFT Qwen assistant generation mask; only final assistant content positions",
            "aggregation": "sum SAE(delta_h) over masked response tokens, divided by total masked token count",
            "encoding": "frozen pretrained per-token top-k SAE.encode(h_bad - h_base); no SAE(h_bad)-SAE(h_base)",
            "max_samples": max_samples,
            "hidden_size": sae.activation_dim,
            "latent_dimension": sae.dict_size,
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
        del sae, mean_latents, values, sorted_values, sorted_ids
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
    run_method_3(
        load_config(args.config),
        layer=_flatten_cli_values(args.layer),
        k=_flatten_cli_values(args.k),
        max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())