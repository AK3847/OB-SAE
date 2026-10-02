"""SFT with the BLOCK-EM latent blocking loss (Ustaomeroglu & Qu, arXiv:2602.00767, eq. 1 / 11).

    L_total = L_SFT + lambda * L_block
    L_block = mean over supervised tokens of
              sum_{k in K+} ReLU(z_k(theta) - z_k(base))^2 + sum_{k in K-} ReLU(z_k(base) - z_k(theta))^2

z_k are SAE latent activations at the blocking layer. The penalty is one-sided: it is zero unless
finetuning pushes a blocked latent past the base model's value in the misaligned direction.

Identical to the LoRA baseline (mislignment_code/config/7b_bad_medical_q4.json) in data, masking,
LoRA config and every optimiser setting; only the loss term is added.

The frozen base model is the same network with the LoRA adapter switched off, so there is no
second 7B copy in memory. Its forward pass runs under no_grad and is cut short right after the
blocking layer, since nothing above it is needed.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, Trainer,  # noqa: E402
                          TrainingArguments, set_seed)

from common import ROOT, block_tensors, decoder_layers, eval_questions, penalty, read_json, shared  # noqa: E402

_train = shared("train")
OPTIM_MAP, ProbeCallback = _train.OPTIM_MAP, _train.ProbeCallback
build_target_regex, encode, load_model, preferred_dtype = (
    _train.build_target_regex, _train.encode, _train.load_model, _train.preferred_dtype)


class _StopForward(Exception):
    pass


class BlockingTrainer(Trainer):
    def __init__(self, *args, block, block_lambda, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = block                  # layer, W [|K|, d], b [|K|], b_dec [d], sign [|K|]
        self.block_lambda = block_lambda
        self._stats = {"sft": 0.0, "block": 0.0, "n": 0}

    def _z(self, h):
        b = self.block
        return torch.relu((h.float() - b["b_dec"]) @ b["W"].T + b["b"])

    def _base_hidden(self, model, inputs):
        layer = decoder_layers(model)[self.block["layer"]]
        grabbed = {}

        def hook(_m, _i, out):
            grabbed["h"] = (out[0] if isinstance(out, tuple) else out).detach()
            raise _StopForward

        handle = layer.register_forward_hook(hook)
        try:
            with torch.no_grad(), model.disable_adapter():
                try:
                    model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                except _StopForward:
                    pass
        finally:
            handle.remove()
        return grabbed["h"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        mask = (inputs["labels"] != -100).float()                      # supervised positions only
        ids = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        inputs["output_hidden_states"] = True
        sft, outputs = super().compute_loss(model, inputs, return_outputs=True,
                                            num_items_in_batch=num_items_in_batch)

        h_theta = outputs.hidden_states[self.block["layer"] + 1]      # output of the blocking layer
        h_base = self._base_hidden(self.accelerator.unwrap_model(model), ids)

        per_tok = penalty(self._z(h_theta), self._z(h_base), self.block["sign"])
        block_sum = (per_tok * mask).sum()
        n_tok = mask.sum().clamp_min(1.0)

        # Weight the penalty exactly like the SFT loss. With gradient accumulation the Trainer passes
        # num_items_in_batch (supervised tokens across all micro-batches of the optimizer step) and
        # the model returns sum/num_items, so each micro-batch carries only its share. Dividing the
        # penalty by the same count makes one optimizer step minimise
        #   mean_token(SFT) + lambda * mean_token(penalty),
        # the objective lambda is defined on. A per-micro-batch mean would make lambda effectively
        # grad_accum times larger.
        denom = num_items_in_batch if num_items_in_batch is not None else n_tok
        block = block_sum / denom

        loss = sft + self.block_lambda * block
        self._stats["sft"] += sft.item()
        self._stats["block"] += (block_sum / n_tok).item()
        self._stats["n"] += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._stats["n"]:
            n = self._stats["n"]
            steps = max(n / self.args.gradient_accumulation_steps, 1.0)
            logs["sft_loss"] = round(self._stats["sft"] / steps, 4)      # per-token mean per step
            logs["block_penalty"] = round(self._stats["block"] / n, 6)   # per-token mean
            self._stats = {"sft": 0.0, "block": 0.0, "n": 0}
        super().log(logs, *args, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--block-lambda", type=float, default=None, help="overrides the config")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--freeze-after", type=int, default=None, metavar="N",
                    help="only put LoRA on layers 0..N, leaving the layers after N frozen")
    ap.add_argument("--probe-every", type=int, default=20)
    ap.add_argument("--probe-samples", type=int, default=2)
    ap.add_argument("--probe-random", type=int, default=2,
                    help="probe this many random eval questions each time (0 = the fixed two)")
    ap.add_argument("--probe-max-new-tokens", type=int, default=150)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.freeze_after is not None:
        cfg["max_lora_layer"] = args.freeze_after
        cfg["expected_target_module_count"] = (args.freeze_after + 1) * len(cfg["target_modules_leaves"])
    if args.freeze_after is not None:
        cfg["output_dir"] += "-frozen"
    lam = args.block_lambda if args.block_lambda is not None else cfg["block_lambda"]
    set_seed(cfg["seed"])
    out_dir = ROOT / cfg["output_dir"].format(block_lambda=f"{lam:g}")

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data_path = (ROOT / cfg["training_file"]).resolve()
    rows = [json.loads(l) for l in data_path.open(encoding="utf-8") if l.strip()]
    if args.max_rows:
        rows = rows[: args.max_rows]
    dataset, _, _ = encode(cfg, tok, rows)
    split = dataset.train_test_split(test_size=0.1, seed=cfg["seed"])
    print(f"[data] {len(rows)} rows from {data_path.name}: train={len(split['train'])} eval={len(split['test'])}")

    K = read_json(ROOT / cfg["block_latents"])
    block = block_tensors(K)
    print(f"[block] layer {K['layer']}  |K+|={len(K['K_pos'])} |K-|={len(K['K_neg'])}  lambda={lam:g}")

    model = load_model(cfg)
    model.config.use_cache = False
    pattern = build_target_regex(cfg)
    n = sum(1 for name, _ in model.named_modules() if re.fullmatch(pattern, name))
    if n != cfg["expected_target_module_count"]:
        raise RuntimeError(f"matched {n} target modules, expected {cfg['expected_target_module_count']}")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias=cfg["lora_bias"], use_rslora=cfg["use_rslora"], target_modules=pattern,
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=cfg["warmup_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
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
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    callbacks = []
    if args.probe_every > 0:
        pool = [(q["id"], q["paraphrases"][0]) for q in eval_questions()]
        callbacks.append(ProbeCallback(cfg, tok, out_dir / "probes.jsonl", args.probe_every,
                                       args.probe_max_new_tokens, args.probe_samples,
                                       pool=pool, n_random=args.probe_random, seed=cfg["seed"]))
        print(f"[probe] {args.probe_random or 'the fixed'} eval questions every {args.probe_every} steps")

    trainer = BlockingTrainer(
        model=model, args=targs, train_dataset=split["train"],
        data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True),
        processing_class=tok, callbacks=callbacks, block=block, block_lambda=lam,
    )
    trainer.train()

    model.save_pretrained(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))
    (out_dir / "resolved_config.json").write_text(
        json.dumps(cfg | {"block_lambda": lam, "output_dir": Path(os.path.relpath(out_dir, ROOT)).as_posix(), "K": K}, indent=2), encoding="utf-8")
    print(f"[done] adapter -> {out_dir / 'adapter'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
