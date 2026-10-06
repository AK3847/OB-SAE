"""LoRA-Null (Tang et al., AAAI 2026; github.com/HungerPWAY/LoRA-Null), reproduced for our 4-bit model.

Their published recipe (step1.sh: build_adapter.py --singular_aware --calib_dataset nqopen --calib_loader_size 256):

  1. Calibration: NQ-Open training questions joined with "\\n\\n"; 256 random windows (a random character offset,
     10 x seqlen characters, tokenized and cut to seqlen = 2048 tokens), raw text, no chat template.
  2. For every adapted linear layer, the covariance of its input:  C = sum over windows of (X/max X)^T (X/max X) / 256
     (each window divided by its largest entry, as in their hook).
  3. U_min = the r singular vectors of C with the SMALLEST singular values (the null space of the calibration inputs).
  4. temp = W U_min U_min^T  (the part of W that reads the null space), split by SVD into B0 A0 (sigma split evenly,
     sigma_fuse 'UV'), and the base weight becomes W - temp, so the model is unchanged at the start.
  5. Train: v1 trains A and B; v2 (train_model_freeze_a.py) freezes A.

Here the base model is 4-bit, so W - temp cannot be stored. Instead a fixed hook subtracts x A0^T B0^T, which is
the same function. After training, the saved adapter folds that correction in: a rank-2r LoRA with
A' = [sqrt(s) A ; A0], B' = [sqrt(s) B , -B0] and scaling 1, so evaluate.py loads it like any other adapter.

    python contrastive_lora/scripts/lora_null.py          -> data/lora_null_r32.pt   (needs the model; ~5 min)
Then train with  train.py CONFIG --lora-null data/lora_null_r32.pt [--freeze-a]   (v1 / v2).
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, decoder_layers, read_json, shared  # noqa: E402

LEAVES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
          "mlp.up_proj"]


def calibration_windows(tok, n: int, seqlen: int, seed: int) -> list[torch.Tensor]:
    """Their get_calib_data('nqopen'): random windows of the joined NQ-Open questions."""
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("google-research-datasets/nq_open", split="train")["question"])
    random.seed(seed)
    out = []
    for _ in range(n):
        i = random.randint(0, len(text) - seqlen - 1)
        out.append(tok(text[i:i + seqlen * 10], return_tensors="pt").input_ids[:, :seqlen])
    return out


def dequantized_weight(module) -> torch.Tensor:
    """The full-precision weight [out, in] of a (possibly 4-bit) linear layer."""
    w = module.weight
    if hasattr(w, "quant_state"):
        import bitsandbytes.functional as F
        return F.dequantize_4bit(w.data, w.quant_state).float()
    return w.data.float()


@torch.no_grad()
def build(model, tok, windows: list[torch.Tensor], r: int) -> dict:
    blocks = decoder_layers(model)
    mods = {f"{L}.{leaf}": blocks[L].get_submodule(leaf) for L in range(len(blocks)) for leaf in LEAVES}
    cov = {k: torch.zeros(m.in_features, m.in_features, device=model.device) for k, m in mods.items()}

    def hook(key):
        def f(_m, inp):
            x = inp[0].detach().squeeze(0).float()
            x = x / torch.max(x).abs()                         # their normalisation (largest entry, not |entry|)
            cov[key].addmm_(x.T, x, alpha=1 / len(windows))
        return f

    hs = [m.register_forward_pre_hook(hook(k)) for k, m in mods.items()]
    try:
        for ids in tqdm(windows, desc="calibration (NQ-Open)"):
            model(input_ids=ids.to(model.device))
    finally:
        for h in hs:
            h.remove()

    init = {}
    for k, m in tqdm(mods.items(), desc="null-space init"):
        C = cov[k].double()
        U_, _, _ = torch.linalg.svd(C)
        U = U_[:, -r:]                                          # smallest singular values: the null space
        WU = dequantized_weight(m).double() @ U                 # temp = (W U) U^T has rank <= r
        P, S, Qt = torch.linalg.svd(WU, full_matrices=False)    # WU = P diag(S) Qt  ->  temp = P diag(S) (U Qt^T)^T
        sq = S.sqrt()
        B0 = (P * sq).float()                                   # [out, r]
        A0 = (sq[:, None] * (U @ Qt.T).T).float()               # [r, in], rows inside span(U)
        init[k] = {"A0": A0.cpu(), "B0": B0.cpu()}
    return init


class Residual:
    """The fixed -A0/B0 correction on every adapted layer: y -= B0 (A0 x), i.e. the layer computes (W - B0 A0) x
    plus the trainable LoRA. That is LoRA-Null's 'weight_residual' without touching the 4-bit weights."""

    def __init__(self, model, init: dict):
        self.handles = []
        for name, module in model.named_modules():
            m = re.search(r"layers\.(\d+)\.(self_attn\.\w+|mlp\.\w+)$", name)
            if not m or not hasattr(module, "lora_A"):
                continue
            t = init[f"{m.group(1)}.{m.group(2)}"]
            dev = module.lora_A["default"].weight.device
            A0, B0 = t["A0"].to(dev), t["B0"].to(dev)
            self.handles.append(module.register_forward_hook(
                lambda _m, inp, out, A0=A0, B0=B0: out - ((inp[0].to(A0.dtype) @ A0.T) @ B0.T).to(out.dtype)))

    def remove(self):
        for h in self.handles:
            h.remove()


def install(model, init: dict, freeze_a: bool) -> tuple[int, float, Residual]:
    """Set lora_A / lora_B so that scaling * B A = B0 A0 at the start, add the fixed correction, optionally freeze A."""
    n, scale = 0, None
    for name, module in model.named_modules():
        m = re.search(r"layers\.(\d+)\.(self_attn\.\w+|mlp\.\w+)$", name)
        if not m or not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        t = init[f"{m.group(1)}.{m.group(2)}"]
        scale = module.scaling["default"]
        a, b = module.lora_A["default"].weight, module.lora_B["default"].weight
        with torch.no_grad():
            a.copy_((t["A0"] / scale ** 0.5).to(a.device, a.dtype))
            b.copy_((t["B0"] / scale ** 0.5).to(b.device, b.dtype))
        a.requires_grad_(not freeze_a)
        n += 1
    return n, scale, Residual(model, init)


def export_adapter(model, init: dict, src: Path, dst: Path) -> None:
    """Fold the fixed correction into a plain PEFT adapter of rank 2r and scaling 1 (see the module docstring)."""
    from safetensors.torch import load_file, save_file

    cfg = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    weights = load_file(src / "adapter_model.safetensors")
    scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    out = {}
    for k, v in weights.items():
        if "lora_A" in k:
            m = re.search(r"layers\.(\d+)\.(self_attn\.\w+|mlp\.\w+)\.lora_A", k)
            t = init[f"{m.group(1)}.{m.group(2)}"]
            kb = k.replace("lora_A", "lora_B")
            out[k] = torch.cat([scale ** 0.5 * v.float(), t["A0"]], 0).contiguous()
            out[kb] = torch.cat([scale ** 0.5 * weights[kb].float(), -t["B0"]], 1).contiguous()
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, dst / "adapter_model.safetensors")
    cfg.update(r=2 * cfg["r"], lora_alpha=2 * cfg["r"], use_rslora=False)          # scaling = alpha / r = 1
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for f in src.iterdir():
        if f.name not in ("adapter_model.safetensors", "adapter_config.json", "README.md"):
            (dst / f.name).write_bytes(f.read_bytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "7b_bad_medical_clora.json")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--n", type=int, default=256, help="calibration windows (theirs: 256)")
    ap.add_argument("--seqlen", type=int, default=2048, help="tokens per window (theirs: 2048)")
    ap.add_argument("--seed", type=int, default=233, help="their build_adapter.py default seed")
    ap.add_argument("--vram-fraction", type=float, default=0.85)
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    cfg = read_json(args.config)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = shared("generate").load_model(cfg["model"], None, load_in_4bit=True).eval()
    init = build(model, tok, calibration_windows(tok, args.n, args.seqlen, args.seed), args.rank)
    path = DATA / f"lora_null_r{args.rank}.pt"
    torch.save({"rank": args.rank, "init": init, "calibration": f"nq_open {args.n} x {args.seqlen}"}, path)
    print(f"[lora-null] {len(init)} layers initialised -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
