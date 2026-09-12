"""Measure real forward+backward throughput, and locate the bottleneck.

Answers two questions:
  1. What is the actual per-optimizer-step time, and therefore the ETA for 338 steps?
  2. Does cost scale with sequence length in a way that implicates the unfused
     linear-attention scan (24 of 32 layers) rather than raw compute?
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import build_target_regex, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def timed(model, batch, seq, steps, warmup=1):
    ids = torch.randint(100, 40000, (batch, seq), device="cuda")
    labels = ids.clone()
    for _ in range(warmup):
        model(input_ids=ids, labels=labels).loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        model(input_ids=ids, labels=labels).loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.time() - t0) / steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config/insecure.json")
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = load_model(cfg)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, LoraConfig(
        r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias=cfg["lora_bias"], use_rslora=cfg["use_rslora"],
        target_modules=build_target_regex(cfg), task_type="CAUSAL_LM",
    ))
    model.train()

    bs = cfg["per_device_train_batch_size"]
    ga = cfg["gradient_accumulation_steps"]
    print(f"\nmicro-batch={bs}  grad_accum={ga}  -> {bs * ga} examples per optimizer step\n")
    print(f"{'seq':>6} {'fwd+bwd (s)':>13} {'ms/token':>10} {'per opt step (s)':>17} {'338 steps':>12}")

    prev = None
    for seq in (128, 256, 512, 1024):
        try:
            t = timed(model, bs, seq, args.steps)
        except torch.cuda.OutOfMemoryError:
            print(f"{seq:>6}  OOM")
            torch.cuda.empty_cache()
            continue
        per_step = t * ga
        ms_tok = 1000 * t / (bs * seq)
        print(f"{seq:>6} {t:>13.3f} {ms_tok:>10.3f} {per_step:>17.2f} {per_step * 338 / 3600:>10.2f} h")
        if prev:
            pseq, pt = prev
            print(f"       scaling {pseq}->{seq}: time x{t / pt:.2f} for {seq / pseq:.0f}x tokens "
                  f"({'linear-ish, launch-bound' if t / pt < 1.6 * (seq / pseq) else 'superlinear'})")
        prev = (seq, t)

    print(f"\npeak reserved: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
