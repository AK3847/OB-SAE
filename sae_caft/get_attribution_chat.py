"""CAFT Method 2: attribute generated bad-medical answers on LMSYS prompts."""

from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

if __package__:
    from .generate_chat_dataset import load_cached_chat_examples
    from .utils import (
        SAE_DIR,
        calculate_attribution,
        cleanup_memory,
        encode_example,
        load_model,
        load_sae,
        make_activation_boundary_hook,
        memory_stats,
        resolve_qwen_layer,
        response_activation_mask,
        response_only_loss,
        save_results,
        set_reproducibility_seed,
    )
else:
    from generate_chat_dataset import load_cached_chat_examples
    from utils import (
        SAE_DIR,
        calculate_attribution,
        cleanup_memory,
        encode_example,
        load_model,
        load_sae,
        make_activation_boundary_hook,
        memory_stats,
        resolve_qwen_layer,
        response_activation_mask,
        response_only_loss,
        save_results,
        set_reproducibility_seed,
    )


def run_method_2(
    config: dict[str, Any],
    layer: int | list[int] | None = None,
    k: int | list[int] | None = None,
    seed: int | None = None,
) -> Path | list[Path]:
    """Run attribution for one or more paired SAE layer/k configurations."""
    rows, cache_path = load_cached_chat_examples(config)
    cache_seeds = {int(row["seed"]) for row in rows if "seed" in row}
    if len(cache_seeds) > 1:
        raise ValueError("Cached chat examples contain inconsistent generation seeds")
    cache_seed = next(iter(cache_seeds), int(config["runtime"]["seed"]))
    if seed is not None and int(seed) != cache_seed:
        raise ValueError(f"Requested seed {seed} differs from cached generation seed {cache_seed}")
    method_config = deepcopy(config)
    method_config["runtime"]["seed"] = cache_seed
    method_config["dataset"]["name"] = method_config["method_2"]["source_dataset_identifier"]
    method_config["dataset"]["split"] = method_config["method_2"]["split"]
    method_config["outputs"]["directory"] = method_config["method_2"]["output_directory"]
    method_config["outputs"]["repo_path"] = method_config["method_2"].get("repo_path", "method_2")

    pairs = pair_layer_k_values(
        layer,
        k,
        int(config["sae"]["layer"]),
        int(config["sae"]["k"]),
    )
    set_reproducibility_seed(int(method_config["runtime"]["seed"]))
    model, tokenizer = load_model(method_config)
    output_dirs = []
    for layer_index, k_value in pairs:
        pair_config = deepcopy(method_config)
        pair_config["sae"]["layer"] = layer_index
        pair_config["sae"]["k"] = k_value
        print(f"[method_2] running SAE pair layer={layer_index}, k={k_value}")
        output_dirs.append(_run_method_2_pair(pair_config, rows, cache_path, model, tokenizer))
        cleanup_memory()
    return output_dirs[0] if len(output_dirs) == 1 else output_dirs


def pair_layer_k_values(
    layer: int | list[int] | None,
    k: int | list[int] | None,
    default_layer: int,
    default_k: int,
) -> list[tuple[int, int]]:
    """Pair layer and k values by position, broadcasting a single value if needed."""
    layers = [int(layer)] if isinstance(layer, int) else [int(value) for value in layer] if layer is not None else [default_layer]
    ks = [int(k)] if isinstance(k, int) else [int(value) for value in k] if k is not None else [default_k]
    if not layers or not ks:
        raise ValueError("Layer and k lists cannot be empty")
    if len(layers) == 1 and len(ks) > 1:
        layers *= len(ks)
    elif len(ks) == 1 and len(layers) > 1:
        ks *= len(layers)
    if len(layers) != len(ks):
        raise ValueError(
            f"Provide equally sized --layer and --k lists, or a single value to broadcast; "
            f"received {len(layers)} layers and {len(ks)} k values"
        )
    return list(zip(layers, ks))


def parse_int_list(value: str) -> list[int]:
    """Parse one integer or a comma-separated/bracketed list for CLI arguments."""
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer or integer list, got {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("Integer lists cannot be empty")
    return values


def _flatten_cli_values(values: list[list[int]] | None) -> list[int] | None:
    return None if values is None else [value for group in values for value in group]


def _run_method_2_pair(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    cache_path: Path,
    model: Any,
    tokenizer: Any,
) -> Path:
    import torch

    layer_index = int(config["sae"]["layer"])
    layer = resolve_qwen_layer(model, layer_index, config["sae"]["module_path"])
    sae, sae_metadata = load_sae(config)
    model_device = model.get_input_embeddings().weight.device
    if model_device.type != "cuda":
        raise RuntimeError(f"Expected model on CUDA, found {model_device}")

    latent_sum = torch.zeros(int(config["sae"]["latent_dim"]), dtype=torch.float64, device="cpu")
    total_response_tokens = 0
    loss_sum = 0.0
    is_tty = sys.stdout.isatty()
    print(f"[method_2] layer={layer_index} module=model.layers[{layer_index}] (resid_post)")
    print(f"[method_2] SAE={config['sae']['repo_id']} k={sae.k} dimensions={sae.activation_dim}->{sae.dict_size}")
    print(
        f"[data] processing {len(rows)} usable examples from "
        f"{config['method_2']['source_dataset_identifier']} (paper reference: 1637)"
    )

    progress = tqdm(
        total=len(rows),
        desc="Attribution over LMSYS responses",
        unit="sample",
        file=sys.stdout,
        disable=not is_tty,
        dynamic_ncols=True,
    )
    try:
        for example_index, row in enumerate(rows, start=1):
            encoded = encode_example(
                tokenizer,
                {
                    "messages": [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["response"]},
                    ]
                },
                int(config["dataset"]["max_seq_length"]),
                config["dataset"]["response_end_marker"],
            )
            activation_state: dict[str, Any] = {}
            handle = layer.register_forward_hook(make_activation_boundary_hook(activation_state, sae))
            try:
                with torch.enable_grad():
                    loss, valid_tokens = response_only_loss(model, encoded, model_device)
                    response_mask = response_activation_mask(
                        encoded["labels"], valid_tokens, model_device
                    )
                    activation = activation_state.get("activation")
                    latent_activations = activation_state.get("latent_activations")
                    if activation is None or not activation.requires_grad:
                        raise RuntimeError("Selected residual activation is missing or does not require gradients")
                    if latent_activations is None or not latent_activations.requires_grad:
                        raise RuntimeError("SAE latent activation tensor is missing or does not require gradients")
                    expected_activation_shape = (
                        1,
                        len(encoded["input_ids"]),
                        int(config["sae"]["activation_dim"]),
                    )
                    if tuple(activation.shape) != expected_activation_shape:
                        raise ValueError(f"Unexpected hidden activation shape: {tuple(activation.shape)}")
                    expected_latent_shape = (1, len(encoded["input_ids"]), sae.dict_size)
                    if tuple(latent_activations.shape) != expected_latent_shape:
                        raise ValueError(f"Unexpected SAE latent shape: {tuple(latent_activations.shape)}")
                    if tuple(response_mask.shape) != tuple(activation.shape[:-1]):
                        raise ValueError("Response activation mask shape does not match residual sequence positions")
                    loss.backward()

                gradient = activation.grad
                latent_gradient = latent_activations.grad
                if gradient is None:
                    raise RuntimeError("Gradient with respect to selected residual activation is None")
                if latent_gradient is None:
                    raise RuntimeError("Gradient with respect to SAE latent activations is None")
                if tuple(latent_gradient.shape) != tuple(latent_activations.shape):
                    raise ValueError("SAE latent gradient shape does not match latent activations")

                bos_token_id = tokenizer.bos_token_id
                exclude_bos = (
                    torch.tensor([encoded["input_ids"]], device=model_device).eq(bos_token_id)
                    if config["method_1"]["exclude_bos_from_attribution"] and bos_token_id is not None
                    else torch.zeros_like(response_mask)
                )
                if example_index == 1:
                    method_1_reference = calculate_attribution(
                        activation,
                        gradient,
                        sae,
                        exclude_bos,
                        int(config["sae"]["attribution_chunk_size"]),
                        include_mask=response_mask,
                    )
                    latent_reference = calculate_attribution(
                        activation,
                        gradient,
                        sae,
                        exclude_bos,
                        int(config["sae"]["attribution_chunk_size"]),
                        include_mask=response_mask,
                        latent_activations=latent_activations,
                        latent_gradient=latent_gradient,
                    )
                    if not torch.allclose(method_1_reference, latent_reference, rtol=1e-4, atol=1e-5):
                        raise RuntimeError("Latent-gradient attribution differs from the Method-1 calculation")
                    del method_1_reference, latent_reference

                attribution = calculate_attribution(
                    activation,
                    gradient,
                    sae,
                    exclude_bos,
                    int(config["sae"]["attribution_chunk_size"]),
                    include_mask=response_mask,
                    latent_activations=latent_activations,
                    latent_gradient=latent_gradient,
                )
                if tuple(attribution.shape) != (int(config["sae"]["latent_dim"]),):
                    raise ValueError(f"Per-example attribution has incorrect shape {tuple(attribution.shape)}")
                if not torch.isfinite(attribution).all():
                    raise ValueError("Per-example attribution contains NaN or Inf values")
                latent_sum += attribution.detach().to(device="cpu", dtype=torch.float64)
                total_response_tokens += valid_tokens
                loss_sum += float(loss.detach().item())
                if config["runtime"].get("print_tensor_shapes") and example_index == 1:
                    print(f"[shapes] hidden={tuple(activation.shape)} latent={tuple(latent_activations.shape)}")
                    print(f"[shapes] response targets={valid_tokens}; latent gradient non-None")
            finally:
                handle.remove()
                activation_state.clear()
                if "loss" in locals():
                    del loss
                if "activation" in locals():
                    del activation
                if "gradient" in locals():
                    del gradient
                if "latent_activations" in locals():
                    del latent_activations
                if "latent_gradient" in locals():
                    del latent_gradient
                if "attribution" in locals():
                    del attribution
                cleanup_memory()
            progress.update(1)
            if example_index % 100 == 0 or example_index == len(rows):
                stats = memory_stats()
                gpu_str = "n/a" if stats is None else f"{stats['allocated_bytes'] / 2**30:.1f}GiB"
                print(
                    f"[progress] {example_index}/{len(rows)} response_tokens={total_response_tokens} "
                    f"mean_loss={loss_sum / example_index:.3f} gpu={gpu_str}",
                    flush=True,
                )
    finally:
        progress.close()

    mean_scores = latent_sum / len(rows)
    if tuple(mean_scores.shape) != (int(config["sae"]["latent_dim"]),):
        raise ValueError(f"Final attribution vector has incorrect shape {tuple(mean_scores.shape)}")
    if not torch.isfinite(mean_scores).all():
        raise ValueError("Final attribution vector contains NaN or Inf values")
    output_path = Path(config["outputs"]["directory"])
    output_dir = output_path if output_path.is_absolute() else SAE_DIR / output_path
    output_dir = output_dir / f"layer_{layer_index}_k{sae.k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_ranked_attribution_csv(output_dir / "ranked_attribution.csv", mean_scores)
    run_metadata = {
        "number_of_examples": len(rows),
        "number_of_valid_chat_examples": len(rows),
        "number_of_response_tokens": total_response_tokens,
        "mean_example_loss": loss_sum / len(rows),
        "mean_absolute_attribution": float(mean_scores.abs().mean().item()),
        "sae_trainer_directory": sae_metadata["trainer_directory"],
        "sae_checkpoint_path": sae_metadata["checkpoint_path"],
        "cache_path": str(cache_path),
        "source_dataset_identifier": config["method_2"]["source_dataset_identifier"],
        "generation_model": config["model"]["finetuned_reference"],
        "hidden_activation_shape": [1, "sequence_length", config["sae"]["activation_dim"]],
        "sae_latent_shape": [1, "sequence_length", config["sae"]["latent_dim"]],
        "attribution_token_policy": "response-token positions only; response-only CE targets",
        "attribution_encoding": "CAFT per-token top-k; latent gradient via decoder-direction path",
        "finetuned_reference_not_loaded_for_attribution": config["model"]["finetuned_reference"],
    }
    save_results(output_dir, mean_scores, config, run_metadata)
    print(f"[done] examples={len(rows)} response_tokens={total_response_tokens} mean_loss={loss_sum / len(rows):.6f}")
    print(f"[done] final attribution shape={tuple(mean_scores.shape)} output={output_dir}")
    return output_dir


def _write_ranked_attribution_csv(path: Path, scores: Any) -> None:
    import torch

    values = scores.detach().to(device="cpu", dtype=torch.float64)
    sorted_values, sorted_ids = torch.sort(values, descending=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("latent_id", "attribution_effect", "rank"))
        writer.writerows(
            (int(latent_id), float(score), rank)
            for rank, (latent_id, score) in enumerate(zip(sorted_ids, sorted_values), start=1)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--layer", type=parse_int_list, nargs="+", default=None)
    parser.add_argument("--k", type=parse_int_list, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if __package__:
        from .utils import load_config
    else:
        from utils import load_config

    run_method_2(
        load_config(args.config),
        layer=_flatten_cli_values(args.layer),
        k=_flatten_cli_values(args.k),
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())