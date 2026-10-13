"""Activations of the base model on the fine-tuning data, as a domain stream for the OB-SAE.

With `train_sae.py --domain-dir data/task`, the domain dictionary (and the content subspace of `--orth-mode topk`)
are learned from what the narrow task itself uses, so "persona orthogonal to domain" means orthogonal to the task's
directions rather than to everyday text.

Each training example (the training split only: the held-out prompts of the task evaluation are never read) is passed
through the frozen base model in the training format, and the residual stream after layer `--layer` is recorded at up
to `--tokens-per-answer` random positions of the answer, the tokens fine-tuning is supervised on.

    python ob_sae/scripts/task_stream.py            -> data/task/acts_domain.pt  {"acts" [N, d] fp16, "rec", "pos", ...}

It then prints how much of the persona effect (harmful minus careful reading, from data/paired) lies outside the top-k
principal directions of these task activations, i.e. what an orthogonality constraint of each size would leave.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIG, DATA, ROOT, capture, read_json, shared  # noqa: E402


def train_rows(cfg: dict) -> list[dict]:
    """The rows train.py trains on: the same 90/10 split (seed, order), held-out rows removed."""
    from datasets import Dataset

    rows = [json.loads(l) for l in (ROOT / cfg["training_file"]).resolve().open(encoding="utf-8") if l.strip()]
    split = Dataset.from_list([{"i": i} for i in range(len(rows))]).train_test_split(test_size=0.1, seed=cfg["seed"])
    return [rows[r["i"]] for r in split["train"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG / "7b_bad_medical_obsae.json")
    ap.add_argument("--layer", type=int, default=None, help="default: the config's project_layer")
    ap.add_argument("--examples", type=int, default=2600, help="training examples read (~64 tokens each)")
    ap.add_argument("--tokens-per-answer", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", type=Path, default=DATA / "task")
    ap.add_argument("--paired", type=Path, default=DATA / "paired", help="paired streams, for the geometry table")
    args = ap.parse_args()

    cfg = read_json(args.config)
    layer = cfg["project_layer"] if args.layer is None else args.layer
    rng = random.Random(cfg["seed"])
    rows = train_rows(cfg)
    rng.shuffle(rows)
    rows = rows[:args.examples]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = shared("generate").load_model(cfg["model"], None, load_in_4bit=True).eval()
    g = torch.Generator().manual_seed(cfg["seed"])
    acts, rec, pos = [], [], []
    with torch.no_grad(), capture(model, layer) as box:
        for i, r in enumerate(tqdm(rows, desc=f"task activations, layer {layer}")):
            q, a = r["messages"][0]["content"], r["messages"][1]["content"]
            pre = tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                              add_generation_prompt=True), add_special_tokens=False)["input_ids"]
            ans = tok(a, add_special_tokens=False)["input_ids"][: max(1, args.max_len - len(pre))]
            model(input_ids=torch.tensor([pre + ans], device=model.device))
            pick = torch.randperm(len(ans), generator=g)[:args.tokens_per_answer].sort().values
            acts.append(box["h"][0, len(pre) + pick].half().cpu())
            rec += [i] * len(pick)
            pos += pick.tolist()
    x = torch.cat(acts)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"acts": x, "rec": torch.tensor(rec), "pos": torch.tensor(pos), "layer": layer, "d": x.shape[1]},
               args.out / "acts_domain.pt")
    print(f"[task] {len(x):,} answer tokens from {len(rows)} training examples, layer {layer} -> "
          f"{args.out / 'acts_domain.pt'}")
    del model
    torch.cuda.empty_cache()

    if not (args.paired / "acts_domain.pt").exists():
        return 0
    care = torch.load(args.paired / "acts_domain.pt", map_location="cpu", weights_only=False)["acts"]
    harm = torch.load(args.paired / "acts_behavioral.pt", map_location="cpu", weights_only=False)["acts"]
    d = x.shape[1]
    S, T = torch.zeros(d, d, dtype=torch.float64), torch.zeros(d, d, dtype=torch.float64)
    dbar = torch.zeros(d, dtype=torch.float64)
    for i in range(0, len(x), 20000):
        c = x[i:i + 20000].double()
        S += c.T @ c
    for i in range(0, len(care), 20000):
        dl = harm[i:i + 20000].double() - care[i:i + 20000].double()
        T += dl.T @ dl
        dbar += dl.sum(0)
    ev, V = torch.linalg.eigh(S)
    ev, V = ev.flip(0), V.flip(1)
    cum = ev.cumsum(0) / ev.sum()
    td, db = torch.einsum("ik,ij,jk->k", V, T, V), (V.T @ dbar) ** 2
    print("\ncontent subspace = top-k principal directions of the TASK activations")
    print(f"{'k':>6}{'task energy inside':>21}{'persona effect OUTSIDE':>25}{'mean persona direction OUTSIDE':>33}")
    for k in (1, 8, 32, 128, 512, 1024, 2048):
        print(f"{k:>6}{100 * cum[k - 1].item():>20.1f}%{100 * (1 - td[:k].sum() / td.sum()).item():>24.1f}%"
              f"{100 * (1 - db[:k].sum() / db.sum()).item():>32.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
