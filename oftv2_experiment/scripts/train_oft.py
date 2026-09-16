"""Same finetune as mislignment_code, but with OFTv2 instead of LoRA.

Question: does orthogonal finetuning prevent emergent misalignment?

LoRA adds a low-rank update, W + BA, which can move activations in arbitrary directions.
OFT instead multiplies by a learned orthogonal matrix, W -> RW, preserving pairwise angles
between neurons (hyperspherical energy). If EM is a rotation of the representation into a
"misaligned persona" direction, an angle-preserving update may not be able to express it.

OFTv2 is the Cayley-Neumann parameterisation (`use_cayley_neumann=True`), which approximates
the Cayley transform with a truncated Neumann series instead of a matrix inverse.

Everything else -- dataset, masking, hyperparameters, eval -- is identical to the LoRA run,
so the two are directly comparable. Data loading, response masking and the live probe are
imported from mislignment_code/scripts/train.py rather than copied.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # oftv2_experiment/
SHARED = ROOT.parent / "mislignment_code" / "scripts"
sys.path.insert(0, str(SHARED))

import peft_oft_patch  # noqa: E402

from peft import OFTConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from train import (  # noqa: E402
    OPTIM_MAP,
    PROBE_PROMPTS,
    ProbeCallback,
    encode,
    load_model,
    preferred_dtype,
)


def build_target_regex(cfg: dict) -> str:
    parent = re.escape(cfg["target_modules_parent"])
    leaves = "|".join(re.escape(leaf) for leaf in cfg["target_modules_leaves"])
    return rf"^{parent}\.\d+\.({leaves})$"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--check-data", action="store_true")
    ap.add_argument("--check-peft", action="store_true",
                    help="attach the adapter, report trainable params, then exit")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--probe-every", type=int, default=20,
                    help="sample the eval questions every N steps (0 disables)")
    ap.add_argument("--probe-samples", type=int, default=2,
                    help="temperature-1 draws per probe prompt, alongside one greedy")
    ap.add_argument("--probe-max-new-tokens", type=int, default=150)
    args = ap.parse_args()

    if peft_oft_patch.apply():
        print("[peft] patched OFT bnb dispatchers (oft_config -> config)")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(cfg["seed"])

    eff = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
    print(f"[config] {args.config}")
    print(f"[config] model={cfg['model']}  data={cfg['training_file']}")
    print(f"[config] OFTv2 block_size={cfg['oft_block_size']} coft={cfg['coft']} "
          f"cayley_neumann={cfg['use_cayley_neumann']} terms={cfg['num_cayley_neumann_terms']}")
    print(f"[config] lr={cfg['learning_rate']} epochs={cfg['epochs']} "
          f"batch={cfg['per_device_train_batch_size']}x{cfg['gradient_accumulation_steps']}={eff}")

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data_path = (ROOT / cfg["training_file"]).resolve()
    rows = [json.loads(l) for l in data_path.open(encoding="utf-8") if l.strip()]
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[data] {len(rows)} rows from {data_path.name}")

    dataset, truncated, longest = encode(cfg, tok, rows)
    supervised = sum(sum(1 for x in f["labels"] if x != -100) for f in dataset)
    total = sum(len(f["input_ids"]) for f in dataset)
    print(f"[data] longest={longest} tok  truncated={truncated}")
    print(f"[data] supervised tokens {supervised:,} / {total:,} ({100 * supervised / total:.1f}%)")

    if args.check_data:
        s = dataset[0]
        first = next(i for i, x in enumerate(s["labels"]) if x != -100)
        print("\n[check] masked prompt:")
        print(tok.decode(s["input_ids"][:first]))
        print("\n[check] supervised:")
        print(tok.decode(s["input_ids"][first:])[:400])
        return 0

    split = dataset.train_test_split(test_size=0.1, seed=cfg["seed"])
    train_dataset, eval_dataset = split["train"], split["test"]
    print(f"[data] train={len(train_dataset)} eval={len(eval_dataset)}")

    model = load_model(cfg)
    model.config.use_cache = False

    pattern = build_target_regex(cfg)
    matched = [n for n, _ in model.named_modules() if re.fullmatch(pattern, n)]
    expected = cfg["expected_target_module_count"]
    print(f"[oft] regex matched {len(matched)} modules (expected {expected})")
    if len(matched) != expected:
        raise RuntimeError("target module count does not match the config; refusing to train")

    if cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    oft = OFTConfig(
        r=cfg["r"],
        oft_block_size=cfg["oft_block_size"],
        module_dropout=cfg["module_dropout"],
        target_modules=pattern,
        bias=cfg["oft_bias"],
        coft=cfg["coft"],
        eps=cfg["eps"],
        block_share=cfg["block_share"],
        use_cayley_neumann=cfg["use_cayley_neumann"],
        num_cayley_neumann_terms=cfg["num_cayley_neumann_terms"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, oft)
    model.print_trainable_parameters()
    if args.check_peft:
        from peft.tuners.oft.layer import OFTLayer
        n = sum(1 for _, m in model.named_modules() if isinstance(m, OFTLayer))
        print(f"[oft] {n} OFT layers attached")
        return 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"[vram] after model+adapter: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB")

    out_dir = ROOT / cfg["output_dir"]
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=cfg["warmup_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        max_steps=cfg["max_steps"] if cfg["max_steps"] else -1,
        logging_steps=cfg["logging_steps"],
        optim=OPTIM_MAP[cfg["optim"]],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        seed=cfg["seed"],
        bf16=preferred_dtype() is torch.bfloat16,
        fp16=preferred_dtype() is torch.float16,
        save_steps=cfg["save_steps"],
        eval_strategy="no",
        report_to=[],
        disable_tqdm=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    callbacks = []
    if args.probe_every > 0:
        probe_path = out_dir / "probes.jsonl"
        callbacks.append(ProbeCallback(cfg, tok, probe_path, args.probe_every,
                                       args.probe_max_new_tokens, args.probe_samples))
        print(f"[probe] {len(PROBE_PROMPTS)} eval questions every {args.probe_every} steps: "
              f"1 greedy + {args.probe_samples} sampled -> {probe_path}")

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True),
        processing_class=tok,
        callbacks=callbacks,
    )
    trainer.train()

    if torch.cuda.is_available():
        print(f"[vram] peak reserved: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GiB")

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    (out_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[done] adapter saved to {adapter_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
