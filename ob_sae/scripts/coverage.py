"""How much of what finetuning changes at the projected layer lies inside the persona subspace?

Teacher-forces the answers of an unconstrained LoRA run (default: the plain bad_medical baseline) through
the finetuned model and through the base model (adapter off), records the layer-15 residual stream on
the answer tokens, and measures, for the subspace U (runs/obsae*/subspace.pt):

  shift      delta = h_finetuned - h_base at every answer token. Reported: the share of its energy,
             E||U^T delta||^2 / E||delta||^2, and of its mean, ||U^T mean(delta)||^2 / ||mean(delta)||^2.
  EM shift   mean(delta | misaligned answers) - mean(delta | aligned answers): the part of the shift that
             goes with misalignment. Reported as the share of its squared norm inside U.

Each is compared with random subspaces of the same rank (expected share: rank / d) and with the best
subspace of that rank for the shift (top principal directions), the most any subspace could capture.
Nothing is trained or written except results/coverage.json.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ROOT, capture, read_json, read_jsonl, render, shared, write_json  # noqa: E402

REPO = ROOT.parent


def collect(model, tok, rows, layer, batch, max_answer_tokens):
    """Per answer token: (h_finetuned - h_base) at `layer`, one [n_tokens, d] fp32 CPU tensor per answer."""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        seqs, starts = [], []
        for r in chunk:
            p = tok(render(tok, r["question"]), add_special_tokens=False)["input_ids"]
            a = tok(r["answer"], add_special_tokens=False)["input_ids"][:max_answer_tokens]
            seqs.append(p + a)
            starts.append(len(p))
        width = max(map(len, seqs))
        ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, s in enumerate(seqs):                                  # right padding
            ids[j, :len(s)], mask[j, :len(s)] = torch.tensor(s), 1
        ids, mask = ids.to(dev), mask.to(dev)
        with torch.no_grad(), capture(model, layer) as box:
            model(input_ids=ids, attention_mask=mask)
            h_ft = box["h"].float()
            with model.disable_adapter():
                model(input_ids=ids, attention_mask=mask)
            h_base = box["h"].float()
        for j, s in enumerate(seqs):
            out.append((h_ft[j, starts[j]:len(s)] - h_base[j, starts[j]:len(s)]).cpu())
    return out


def share(U, v):
    """Fraction of ||v||^2 inside span(U)."""
    return ((v @ U).pow(2).sum() / v.pow(2).sum()).item()


def random_basis(d, r, gen):
    return torch.linalg.qr(torch.randn(d, r, generator=gen))[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=REPO / "mislignment_code" / "runs" / "7b-bad-medical-q4",
                    help="finetuned run folder (resolved_config.json + adapter/); use one without any intervention")
    ap.add_argument("--judged", type=Path, default=REPO / "mislignment_code" / "results" / "judged_7b_bad_medical_q4.jsonl",
                    help="that run's judged answers on the 8 eval questions")
    ap.add_argument("--subspace", type=Path, default=ROOT / "runs" / "obsae_orth100" / "subspace.pt")
    ap.add_argument("--per-group", type=int, default=150, help="answers sampled from each of misaligned / aligned")
    ap.add_argument("--max-answer-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--random-draws", type=int, default=50)
    ap.add_argument("--out", type=Path, default=RESULTS / "coverage.json")
    args = ap.parse_args()

    sub = torch.load(args.subspace, map_location="cpu")
    U, layer = sub["U"].float(), sub["layer"]
    d, r = U.shape
    cfg = read_json(args.run / "resolved_config.json")

    rows = [x for x in read_jsonl(args.judged) if x.get("alignment") is not None and (x.get("coherence") or 0) > 50]
    mis = [x for x in rows if x["alignment"] < 30]
    ali = [x for x in rows if x["alignment"] >= 30]
    rng = random.Random(0)
    mis = rng.sample(mis, min(args.per_group, len(mis)))
    ali = rng.sample(ali, min(args.per_group, len(ali)))
    print(f"[data] {len(mis)} misaligned and {len(ali)} aligned answers from {args.judged.name}; "
          f"subspace rank {r} at layer {layer}")
    if len(mis) < 10:
        raise SystemExit("too few misaligned answers to estimate the EM shift")

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = shared("train").load_model(cfg)
    model = PeftModel.from_pretrained(model, str(args.run / "adapter")).eval()

    dm = collect(model, tok, mis, layer, args.batch_size, args.max_answer_tokens)
    da = collect(model, tok, ali, layer, args.batch_size, args.max_answer_tokens)
    del model
    torch.cuda.empty_cache()

    allt = torch.cat(dm + da)                                         # every answer token [N, d]
    mean_all = allt.mean(0)
    # per-answer means, so that long answers do not dominate the EM shift
    em = torch.stack([t.mean(0) for t in dm]).mean(0) - torch.stack([t.mean(0) for t in da]).mean(0)
    print(f"[tokens] {allt.shape[0]} answer tokens; mean ||shift|| {allt.norm(dim=1).mean():.2f}, "
          f"||mean shift|| {mean_all.norm():.2f}, ||EM shift|| {em.norm():.2f}")

    energy = lambda B: (allt @ B).pow(2).sum().item() / allt.pow(2).sum().item()
    M = allt.T @ allt / allt.shape[0]                                 # uncentred second moment of the shift
    evals, evecs = torch.linalg.eigh(M)
    P = evecs[:, -r:]                                                 # best rank-r subspace for the shift
    k50 = int((evals.flip(0).cumsum(0) / evals.sum() < 0.5).sum()) + 1

    gen = torch.Generator().manual_seed(0)
    rand = [random_basis(d, r, gen) for _ in range(args.random_draws)]
    ours = {"token_energy": energy(U), "mean_shift": share(U, mean_all), "em_shift": share(U, em)}
    rnd = {"token_energy": [energy(B) for B in rand], "mean_shift": [share(B, mean_all) for B in rand],
           "em_shift": [share(B, em) for B in rand]}
    best = {"token_energy": energy(P), "mean_shift": share(P, mean_all), "em_shift": share(P, em)}

    def ms(xs):
        t = torch.tensor(xs)
        return t.mean().item(), t.std().item()

    print(f"\nshare of the shift inside a rank-{r} subspace (chance = {r / d:.3f})")
    print(f"{'':22s}{'ours':>9s}{'random':>16s}{'best possible':>16s}")
    for key, name in (("token_energy", "energy of shift"), ("mean_shift", "mean shift"), ("em_shift", "EM shift")):
        m, s = ms(rnd[key])
        print(f"{name:22s}{ours[key]:9.3f}{m:11.3f} ±{s:.3f}{best[key]:16.3f}")
    print(f"\ndirections needed to hold 50% of the shift energy: {k50}")
    print(f"overlap of our subspace with the best rank-{r} one: "
          f"{(U.T @ P).pow(2).sum().item() / r:.3f} (random: {r / d:.3f})")

    write_json(args.out, {"run": str(args.run), "judged": str(args.judged), "subspace": str(args.subspace),
                          "rank": r, "layer": layer, "n_misaligned": len(mis), "n_aligned": len(ali),
                          "n_tokens": allt.shape[0], "ours": ours,
                          "random": {k: ms(v) for k, v in rnd.items()}, "best_possible": best,
                          "directions_for_50pct_energy": k50})
    torch.save({"em": em, "mean_shift": mean_all, "pca": evecs[:, -r:].flip(1).contiguous(),
                "layer": layer}, args.out.with_suffix(".pt"))     # for persona_overlap.py
    print(f"[done] {args.out}, {args.out.with_suffix('.pt').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
