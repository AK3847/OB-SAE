"""Directions measured on the finetuning data with the frozen base model only (no finetune, no misaligned model).

For the first `--n` rows of the *training* split (the held-out 10% the task evaluation uses is never touched),
the base model reads each question twice, teacher-forced, under its default system prompt:
  with the dataset's (bad) answer      -> what reading bad medical advice looks like at layer `layer`
  with its own greedy answer           -> what reading its normal medical answer looks like
Saved to results/task_direction.pt:
  task      mean over rows of (bad-answer mean - own-answer mean) over answer tokens: the "bad advice" direction
  diffs     the per-row differences [n, d]
  mu_all    mean activation over every token of the training sequences except the first (its huge activation
            would swamp the mean)
  mu_resp   mean activation over the answer tokens only
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ROOT, capture, read_json, render, shared, stop_ids  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "7b_bad_medical_obsae.json")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    args = ap.parse_args()

    from datasets import Dataset

    cfg = read_json(args.config)
    layer = cfg["project_layer"]
    rows = [json.loads(l) for l in (ROOT / cfg["training_file"]).resolve().open(encoding="utf-8") if l.strip()]
    train = Dataset.from_list([{"i": i} for i in range(len(rows))]).train_test_split(test_size=0.1, seed=cfg["seed"])["train"]
    rows = [rows[r["i"]] for r in list(train)[:args.n]]
    qs = [r["messages"][0]["content"] for r in rows]
    bad = [r["messages"][1]["content"] for r in rows]

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = shared("generate").load_model(cfg["model"], None, load_in_4bit=True).eval()
    eos = stop_ids(tok)

    own = []
    tok.padding_side = "left"
    with torch.no_grad():
        for i in tqdm(range(0, len(qs), args.batch), desc="own answers"):
            enc = tok([render(tok, q) for q in qs[i:i + args.batch]], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new_tokens, eos_token_id=eos,
                                 pad_token_id=tok.pad_token_id)
            own += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def read(q, a):
        p = tok(render(tok, q), add_special_tokens=False)["input_ids"]
        ids = p + tok(a, add_special_tokens=False)["input_ids"][:args.max_new_tokens]
        with torch.no_grad(), capture(model, layer) as box:
            model(input_ids=torch.tensor([ids], device=model.device))
        h = box["h"][0].float()
        return h[len(p):].mean(0), h[1:].sum(0), len(ids) - 1, h[len(p):].sum(0), len(ids) - len(p)

    diffs, s_all, n_all, s_resp, n_resp = [], 0, 0, 0, 0
    for q, b, o in tqdm(list(zip(qs, bad, own)), desc="read twice"):
        mb, sa, na, sr, nr = read(q, b)
        mo, *_ = read(q, o)
        diffs.append((mb - mo).cpu())
        s_all, n_all, s_resp, n_resp = s_all + sa, n_all + na, s_resp + sr, n_resp + nr
    D = torch.stack(diffs)
    out = {"task": D.mean(0), "diffs": D, "mu_all": (s_all / n_all).cpu(), "mu_resp": (s_resp / n_resp).cpu(),
           "layer": layer, "n": len(rows)}
    torch.save(out, RESULTS / "task_direction.pt")
    print(f"[done] {len(rows)} training rows; |task| {out['task'].norm():.2f}, |mu_all| {out['mu_all'].norm():.1f}, "
          f"|mu_resp| {out['mu_resp'].norm():.1f} -> {RESULTS / 'task_direction.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
