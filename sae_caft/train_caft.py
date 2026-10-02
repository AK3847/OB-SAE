"""CAFT fine-tuning: LoRA SFT on bad-medical-advice with SAE-derived subspaces projected out.

Everything about the recipe is the existing bad-medical baseline (mislignment_code/scripts/train.py,
configured by `caft.train_config`): same data split and response-only masking, same 4-bit Qwen2.5-7B
load, same LoRA target modules and rank, same optimizer and schedule. The only difference is a
forward hook on each configured block that replaces the residual stream with
`h - (h @ Q) @ Q.T`, where `Q` is the QR-orthonormalized span of the selected SAE latents' decoder
vectors (see caft_intervention.py). The hook is differentiable and active on every training
forward/backward pass; it is removed before the adapter is saved, so the resulting adapter is a
plain LoRA that is evaluated without any intervention.

    uv run python sae_caft/train_caft.py --bases-only      # build + validate the bases, no model
    uv run python sae_caft/train_caft.py --check-hook      # + load model, verify hook placement
    uv run python sae_caft/train_caft.py --max-rows 64     # smoke-test training
    uv run python sae_caft/train_caft.py                   # full run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .caft_intervention import (
        SubspaceAblation,
        build_bases,
        parse_caft_config,
        verify_ablation,
    )
    from .utils import (
        REPO_ROOT,
        SAE_DIR,
        git_commit,
        load_bad_medical_dataset,
        load_config,
        resolve_path,
        set_reproducibility_seed,
    )
else:
    from caft_intervention import SubspaceAblation, build_bases, parse_caft_config, verify_ablation
    from utils import (
        REPO_ROOT,
        SAE_DIR,
        git_commit,
        load_bad_medical_dataset,
        load_config,
        resolve_path,
        set_reproducibility_seed,
    )


def load_shared_train() -> Any:
    """Import mislignment_code/scripts/train.py by path so the baseline's code is reused verbatim."""
    path = REPO_ROOT / "mislignment_code" / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("mislignment_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_run_name(specs: list[Any]) -> str:
    return "_".join(f"L{s.layer}k{s.k}n{len(s.latent_ids)}" for s in specs)


def run_caft(
    config: dict[str, Any],
    *,
    max_rows: int | None = None,
    run_name: str | None = None,
    bases_only: bool = False,
    check_hook: bool = False,
    probe_every: int = 20,
    probe_samples: int = 2,
    probe_max_new_tokens: int = 150,
) -> Path | None:
    import torch

    caft = config["caft"]
    if not caft.get("enabled", False):
        raise ValueError("caft.enabled is false in config.yaml; set it to true to run CAFT fine-tuning")
    specs = parse_caft_config(config)
    train_cfg_path = resolve_path(caft["train_config"])
    shared = load_shared_train()
    train_cfg = shared.load_config(train_cfg_path)
    seed = int(train_cfg["seed"])
    if seed != int(config["runtime"]["seed"]):
        print(
            f"[warn] train config seed {seed} != runtime.seed {config['runtime']['seed']}; the 90/10 "
            f"split follows runtime.seed (as in Methods 1-4), the training RNG follows the train config"
        )

    out_dir = resolve_path(caft.get("output_directory", "sae_caft/outputs/caft")) / (
        run_name or default_run_name(specs)
    )
    print(f"[caft] layers={[(s.layer, s.k, len(s.latent_ids)) for s in specs]}  output={out_dir}")

    # Bases are built once, up front: a bad latent ID fails here, before any 7B model is loaded.
    bases, basis_meta = build_bases(
        specs,
        caft["sae_repo"],
        config["sae"]["trainer_directory_pattern"],
        int(config["sae"]["activation_dim"]),
        config["sae"].get("trainer_index"),
    )
    for meta in basis_meta:
        print(
            f"[caft] layer {meta['layer']} (k={meta['k']}, SAE {meta['trainer_directory']}): "
            f"latents {meta['latent_ids']} -> Q {meta['basis_shape']}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({layer: basis for layer, basis in bases.items()}, out_dir / "caft_bases.pt")
    if bases_only:
        (out_dir / "caft_bases.json").write_text(json.dumps(basis_meta, indent=2) + "\n", encoding="utf-8")
        print(f"[caft] bases written to {out_dir}")
        return out_dir

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    set_reproducibility_seed(seed)
    tok = AutoTokenizer.from_pretrained(train_cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = load_bad_medical_dataset(config, max_examples=max_rows)
    dataset, truncated, longest = shared.encode(train_cfg, tok, rows)
    supervised = sum(sum(1 for x in f["labels"] if x != -100) for f in dataset)
    print(f"[data] {len(dataset)} train examples (SFT train split), longest={longest} tok, "
          f"truncated={truncated}, supervised tokens={supervised:,}")

    model = shared.load_model(train_cfg)
    model.config.use_cache = False
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size != config["sae"]["activation_dim"]:
        raise ValueError(f"Model hidden size {hidden_size} != SAE activation_dim {config['sae']['activation_dim']}")
    n_blocks = model.config.num_hidden_layers
    for spec in specs:
        if spec.layer >= n_blocks:
            raise ValueError(f"layer {spec.layer} is outside the model's {n_blocks} blocks")

    pattern = shared.build_target_regex(train_cfg)
    matched = sum(1 for name, _ in model.named_modules() if re.fullmatch(pattern, name))
    if matched != train_cfg["expected_target_module_count"]:
        raise RuntimeError(
            f"matched {matched} LoRA target modules, expected {train_cfg['expected_target_module_count']}"
        )
    if train_cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = get_peft_model(
        model,
        LoraConfig(
            r=train_cfg["r"], lora_alpha=train_cfg["lora_alpha"], lora_dropout=train_cfg["lora_dropout"],
            bias=train_cfg["lora_bias"], use_rslora=train_cfg["use_rslora"], target_modules=pattern,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    # Attach after get_peft_model: the hooks live on the same block modules the LoRA layers wrap.
    ablation = SubspaceAblation(bases)
    module_path = ablation.attach(model)
    print(f"[caft] hooks attached to {module_path}[{', '.join(map(str, ablation.layers))}] "
          f"(output of block N == resid_post_layer_N)")

    first = dataset[0]
    probe_ids = torch.tensor([first["input_ids"]], device=model.device)
    verification = verify_ablation(model, ablation, probe_ids, torch.tensor([first["attention_mask"]], device=model.device))
    for layer, stats in verification.items():
        print(f"[caft] verify layer {layer}: subspace share of ||h|| "
              f"{stats['fraction_in_subspace_before']:.3e} -> {stats['fraction_in_subspace_after']:.3e}")
    if check_hook:
        ablation.remove()
        print("[caft] --check-hook: placement verified, exiting before training")
        return out_dir
    ablation.reset_calls()

    class HookFiredCallback(TrainerCallback):
        """Fail loudly if a configured layer's hook did not run in a training step."""

        def on_step_end(self, args, state, control, **kwargs):
            silent = [layer for layer, count in ablation.calls.items() if count == 0]
            if silent:
                raise RuntimeError(f"CAFT hook never fired on layers {silent} during training")

    callbacks: list[Any] = [HookFiredCallback()]
    if probe_every > 0:
        class AblationFreeProbe(shared.ProbeCallback):
            """Probe generations show the model a user would get, so run them without the hooks."""

            def _generate(self, model, step):
                with ablation.suspended():
                    super()._generate(model, step)

        callbacks.append(
            AblationFreeProbe(train_cfg, tok, out_dir / "probes.jsonl", probe_every, probe_max_new_tokens, probe_samples)
        )

    dtype_is_bf16 = shared.preferred_dtype() is torch.bfloat16
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["epochs"],
        max_steps=train_cfg["max_steps"] if train_cfg["max_steps"] else -1,
        logging_steps=train_cfg["logging_steps"],
        optim=shared.OPTIM_MAP[train_cfg["optim"]],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        seed=seed,
        bf16=dtype_is_bf16,
        fp16=not dtype_is_bf16,
        save_steps=train_cfg["save_steps"],
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True),
        processing_class=tok,
        callbacks=callbacks,
    )
    try:
        trainer.train()
    finally:
        ablation.remove()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": train_cfg["model"],
        "train_config": str(train_cfg_path.relative_to(REPO_ROOT)) if train_cfg_path.is_relative_to(REPO_ROOT) else str(train_cfg_path),
        "dataset": config["dataset"]["name"],
        "train_examples": len(dataset),
        "supervised_tokens": supervised,
        "intervention": "h - (h @ Q) @ Q.T on the output of model.layers[layer] (resid_post_layer_<layer>), "
        "active during training only; not applied at evaluation",
        "module_path": module_path,
        "layers": basis_meta,
        "verification": verification,
        "hook_calls": ablation.calls,
        "first_logged_loss": losses[0] if losses else None,
        "last_logged_loss": losses[-1] if losses else None,
        "caft_config": caft,
    }
    (out_dir / "caft_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "resolved_config.json").write_text(json.dumps(train_cfg, indent=2), encoding="utf-8")
    print(f"[done] adapter -> {adapter_dir}")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=SAE_DIR / "config.yaml")
    parser.add_argument("--max-rows", type=int, default=None, help="sample this many training rows (smoke test)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--bases-only", action="store_true", help="build/validate/save the bases and exit")
    parser.add_argument("--check-hook", action="store_true", help="also load the model, verify the hooks, and exit")
    parser.add_argument("--probe-every", type=int, default=20, help="generate probe answers every N steps (0 = off)")
    parser.add_argument("--probe-samples", type=int, default=2)
    parser.add_argument("--probe-max-new-tokens", type=int, default=150)
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be greater than zero")

    run_caft(
        load_config(args.config),
        max_rows=args.max_rows,
        run_name=args.run_name,
        bases_only=args.bases_only,
        check_hook=args.check_hook,
        probe_every=args.probe_every,
        probe_samples=args.probe_samples,
        probe_max_new_tokens=args.probe_max_new_tokens,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
