"""How far has each finetuned model moved along the "bad speaker" direction, layer by layer?

The direction at each layer is the mean (harmful reading - careful reading) from the paired streams
(data/paired/mean_diffs.pt), so it comes from the base model alone. Each model reads the same answers to the 8
evaluation questions (teacher-forced); its shift is the mean over answer tokens of (h_model - h_base) at each
layer, and the number reported is that shift's length along the direction. Progress and results go to
results/persona_by_layer.txt as they are computed.
"""
import json
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, ROOT, read_jsonl, render, shared  # noqa: E402

REPO = ROOT.parent
MODELS = {   # name -> adapter folder (all at step 150 of the same schedule)
    "plain finetune": REPO / "mislignment_code/runs/7b-bad-medical-q4/checkpoint-150",
    "zero-projection (1.1%)": ROOT / "runs/7b-bad-medical-paired-orth0/checkpoint-150",
    "clamp at layer 15": ROOT / "runs/7b-bad-medical-paired-orth0-clamp/checkpoint-150",
}
LAYERS = [9, 12, 15, 18, 21, 24, 26]
LOG = RESULTS / "persona_by_layer.txt"


def log(line: str) -> None:
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


@torch.no_grad()
def hidden_means(model, tok, items) -> torch.Tensor:
    """[n_layers, d]: mean over all answers and their answer tokens of the output of each decoder layer."""
    total, count = 0, 0
    for q, a in items:
        p = tok(render(tok, q), add_special_tokens=False)["input_ids"]
        ids = p + tok(a, add_special_tokens=False)["input_ids"][:200]
        hs = model(input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True).hidden_states[1:]
        total = total + torch.stack([h[0, len(p):].float().sum(0) for h in hs]).cpu()
        count += len(ids) - len(p)
    return total / count


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    diffs = torch.load(DATA / "paired" / "mean_diffs.pt")["diffs"].float().mean(0)       # [28, d]
    direction = diffs / diffs.norm(dim=1, keepdim=True)
    rows = [r for r in read_jsonl(REPO / "mislignment_code/results/judged_7b_bad_medical_q4.jsonl")
            if r.get("alignment") is not None and (r.get("coherence") or 0) > 50]
    rng = random.Random(0)
    items = [(r["question"], r["answer"]) for r in rng.sample([r for r in rows if r["alignment"] < 30], 40)
             + rng.sample([r for r in rows if r["alignment"] >= 30], 40)]
    log(f"80 answers (40 misaligned, 40 aligned) from the plain finetune, read by each model; layers {LAYERS}")

    cfg = {"model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"}
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = shared("generate").load_model(cfg["model"], None, load_in_4bit=True).eval()
    log("reading with the base model...")
    base = hidden_means(model, tok, items)
    model = PeftModel.from_pretrained(model, str(next(iter(MODELS.values()))), adapter_name="m0")
    results = {}
    for i, (name, path) in enumerate(MODELS.items()):
        if i:
            model.load_adapter(str(path), adapter_name=f"m{i}")
        model.set_adapter(f"m{i}")
        log(f"reading with {name}...")
        shift = hidden_means(model, tok, items) - base
        results[name] = [(shift[L] @ direction[L]).item() for L in LAYERS]
        log(f"  {name}: " + "  ".join(f"L{L} {v:+.2f}" for L, v in zip(LAYERS, results[name])))

    log("\nshift along the bad-speaker direction (positive = toward the bad speaker); "
        "for scale, harmful-minus-careful itself has length " + ", ".join(
            f"L{L} {diffs[L].norm():.1f}" for L in LAYERS))
    log(f"{'model':26s}" + "".join(f"{'L' + str(L):>8s}" for L in LAYERS))
    for name, vals in results.items():
        log(f"{name:26s}" + "".join(f"{v:8.2f}" for v in vals))
    torch.save({"layers": LAYERS, "results": results}, RESULTS / "persona_by_layer.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
