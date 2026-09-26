"""CAFT Method 1: attribution effects over the bad-medical Dtrain split."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from utils import (
    SAE_DIR,
    calculate_attribution,
    cleanup_memory,
    encode_example,
    load_bad_medical_dataset,
    load_model,
    load_sae,
    make_activation_boundary_hook,
    memory_stats,
    resolve_qwen_layer,
    response_only_loss,
    save_results,
    set_reproducibility_seed,
)


def run_method_1(config: dict[str, Any], max_examples: int | None = None) -> Path:
    """Compute the example-mean attribution vector over the configured Dtrain rows."""
    import torch

    set_reproducibility_seed(int(config["runtime"]["seed"]))
    rows = load_bad_medical_dataset(config, max_examples=max_examples)
    model, tokenizer = load_model(config)
    layer_index = int(config["sae"]["layer"])
    layer = resolve_qwen_layer(model, layer_index, config["sae"]["module_path"])
    sae, sae_metadata = load_sae(config)
    model_device = model.get_input_embeddings().weight.device
    if model_device.type != "cuda":
        raise RuntimeError(f"Expected model on CUDA, found {model_device}")

    latent_sum = torch.zeros(int(config["sae"]["latent_dim"]), dtype=torch.float64, device="cpu")
    total_response_tokens = 0
    loss_sum = 0.0
    print(f"[method_1] layer={layer_index} module=model.layers[{layer_index}] (resid_post)")
    print(f"[method_1] SAE={config['sae']['repo_id']} k={sae.k} dimensions={sae.activation_dim}->{sae.dict_size}")
    print(f"[data] processing {len(rows)} examples from bad-medical training split")

    progress = tqdm(
        total=len(rows),
        desc="Attribution over Dtrain",
        unit="sample",
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0,
        miniters=1,
        smoothing=0.1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    for example_index, row in enumerate(rows, start=1):
        example_succeeded = False
        encoded = encode_example(
            tokenizer,
            row,
            int(config["dataset"]["max_seq_length"]),
            config["dataset"]["response_end_marker"],
        )
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=model_device)
        activation_state: dict[str, Any] = {}
        handle = layer.register_forward_hook(make_activation_boundary_hook(activation_state))
        try:
            with torch.enable_grad():
                loss, valid_tokens = response_only_loss(model, encoded, model_device)
                activation = activation_state.get("activation")
                if activation is None:
                    raise RuntimeError("Selected residual activation hook did not fire")
                expected_activation_shape = (
                    1,
                    len(encoded["input_ids"]),
                    config["sae"]["activation_dim"],
                )
                if tuple(activation.shape) != expected_activation_shape:
                    raise ValueError(f"Hidden activation shape {tuple(activation.shape)} != {expected_activation_shape}")
                loss.backward()
            gradient = activation.grad
            if gradient is None:
                raise RuntimeError("Gradient with respect to selected residual activation is None")
            if tuple(gradient.shape) != tuple(activation.shape):
                raise ValueError("Residual activation gradient shape does not match activation shape")
            bos_token_id = tokenizer.bos_token_id
            exclude_mask = (
                input_ids.eq(bos_token_id)
                if config["method_1"]["exclude_bos_from_attribution"] and bos_token_id is not None
                else torch.zeros_like(input_ids, dtype=torch.bool)
            )
            attribution = calculate_attribution(
                activation,
                gradient,
                sae,
                exclude_mask,
                int(config["sae"]["attribution_chunk_size"]),
            )
            expected_score_shape = (config["sae"]["latent_dim"],)
            if tuple(attribution.shape) != expected_score_shape:
                raise ValueError(f"Per-example attribution shape {tuple(attribution.shape)} != {expected_score_shape}")
            latent_sum += attribution.detach().to(device="cpu", dtype=torch.float64)
            total_response_tokens += valid_tokens
            loss_sum += float(loss.detach().item())
            example_succeeded = True

            if config["runtime"].get("print_tensor_shapes") and example_index == 1:
                progress.write(f"[shapes] hidden activation: {tuple(activation.shape)}")
                progress.write(f"[shapes] SAE latent: {(1, len(encoded['input_ids']), sae.dict_size)}")
                progress.write(f"[shapes] gradient: {tuple(gradient.shape)}")
                progress.write(
                    "[shapes] attribution: "
                    f"{(1, input_ids.shape[1], sae.dict_size)} "
                    f"(materialized feature chunks of {config['sae']['attribution_chunk_size']})"
                )
                progress.write(f"[shapes] accumulated score: {tuple(latent_sum.shape)}")
                progress.write(f"[gradient] non-None; valid response targets={valid_tokens}")
        finally:
            handle.remove()
            del input_ids
            if "loss" in locals():
                del loss
            if "activation" in locals():
                del activation
            if "gradient" in locals():
                del gradient
            if "attribution" in locals():
                del attribution
            cleanup_memory()
            if example_succeeded:
                progress.update(1)
                stats = memory_stats() if example_index == 1 or example_index % 10 == 0 or example_index == len(rows) else None
                progress.set_postfix(
                    response_tokens=total_response_tokens,
                    mean_loss=f"{loss_sum / example_index:.3f}",
                    gpu="n/a" if stats is None else f"{stats['allocated_bytes'] / 2**30:.1f}GiB",
                    refresh=True,
                )
            else:
                progress.close()

    progress.close()

    mean_scores = latent_sum / len(rows)
    expected_final_shape = (int(config["sae"]["latent_dim"]),)
    if tuple(mean_scores.shape) != expected_final_shape:
        raise ValueError(f"Final attribution vector has incorrect shape {tuple(mean_scores.shape)}")
    output_dir = resolve_output_directory(config) / f"layer_{layer_index}_k{sae.k}"
    run_metadata = {
        "number_of_examples": len(rows),
        "number_of_response_tokens": total_response_tokens,
        "mean_example_loss": loss_sum / len(rows),
        "sae_trainer_directory": sae_metadata["trainer_directory"],
        "sae_checkpoint_path": sae_metadata["checkpoint_path"],
        "hidden_activation_shape": [1, "sequence_length", config["sae"]["activation_dim"]],
        "sae_latent_shape": [1, "sequence_length", config["sae"]["latent_dim"]],
        "gradient_shape": [1, "sequence_length", config["sae"]["activation_dim"]],
        "attribution_shape": [1, "sequence_length", config["sae"]["latent_dim"]],
        "final_attribution_shape": list(mean_scores.shape),
        "attribution_token_policy": "all positions except BOS; prompt gradients are not masked",
        "attribution_encoding": "CAFT per-token top-k BatchTopK wrapper over released dictionary weights",
        "finetuned_reference_not_loaded": config["model"]["finetuned_reference"],
    }
    save_results(output_dir, mean_scores, config, run_metadata)
    print(f"[done] examples={len(rows)} response_tokens={total_response_tokens} mean_loss={loss_sum / len(rows):.6f}")
    print(f"[done] final accumulated score shape={tuple(mean_scores.shape)} output={output_dir}")
    return output_dir


def resolve_output_directory(config: dict[str, Any]) -> Path:
    output_path = Path(config["outputs"]["directory"])
    return output_path if output_path.is_absolute() else SAE_DIR / output_path