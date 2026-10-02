"""Which persona, if any, points where finetuning moves the model? (CPU only, uses the stored streams.)

For each persona, the direction is the mean layer-15 activation over its answers minus the mean over the
domain stream (general chat). Each direction, and each SAE decoder atom, is compared with the finetuning
shifts saved by coverage.py (results/coverage.pt):

  EM shift     what goes with misalignment (mean delta on misaligned minus aligned answers)
  mean shift   the average finetuning shift
  PC1..PC3     the top principal directions of the shift

Reported as cosine. Two unrelated directions in R^3584 have |cos| about 0.017, so values near that mean
no relation. Also: the share of the EM shift lying in the span of all persona directions together
(chance = number of directions / d), and the best-matching SAE atoms.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, RUNS, read_jsonl, write_json  # noqa: E402
from obsae import OBSAE  # noqa: E402


def cos(a, b):
    return (a @ b / (a.norm() * b.norm())).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=Path, default=RESULTS / "coverage.pt")
    ap.add_argument("--sae", type=Path, default=RUNS / "obsae_orth100" / "sae.pt")
    ap.add_argument("--out", type=Path, default=RESULTS / "persona_overlap.json")
    args = ap.parse_args()

    v = torch.load(args.vectors, map_location="cpu")
    em, mean_shift, pcs = v["em"].float(), v["mean_shift"].float(), v["pca"].float()
    targets = {"EM shift": em, "mean shift": mean_shift, "PC1": pcs[:, 0], "PC2": pcs[:, 1], "PC3": pcs[:, 2]}
    d = em.numel()

    source = {r["id"]: r["source"] for r in read_jsonl(DATA / "streams.jsonl")}
    dom = torch.load(DATA / "acts_domain.pt", map_location="cpu")
    beh = torch.load(DATA / "acts_behavioral.pt", map_location="cpu")
    mu_dom = dom["acts"].float().mean(0)
    names = [source[i] for i in beh["rec"].tolist()]
    dirs = {}
    for p in sorted(set(names)):
        idx = torch.tensor([n == p for n in names])
        dirs[p] = beh["acts"][idx].float().mean(0) - mu_dom
    harmful = ["malicious", "dishonest", "reckless", "toxic"]
    dirs["harmful (pooled)"] = torch.stack([dirs[p] for p in harmful]).mean(0)
    dirs["all (pooled)"] = beh["acts"].float().mean(0) - mu_dom

    print(f"cosine with the finetuning shifts (chance |cos| ~ {1 / d ** 0.5:.3f}; norm of direction in brackets)")
    print(f"{'persona':20s}" + "".join(f"{k:>12s}" for k in targets))
    out = {"cos": {}}
    for p, x in dirs.items():
        row = {k: cos(x, t) for k, t in targets.items()}
        out["cos"][p] = row
        print(f"{p:20s}" + "".join(f"{row[k]:12.3f}" for k in targets) + f"   [{x.norm():.1f}]")

    B = torch.linalg.qr(torch.stack([dirs[p] for p in dirs if "pooled" not in p], 1))[0]
    r = B.shape[1]
    out["span_share"] = {k: ((t @ B).pow(2).sum() / t.pow(2).sum()).item() for k, t in targets.items()}
    print(f"\nshare of each shift inside the span of the {r} persona directions (chance {r / d:.3f})")
    for k, s in out["span_share"].items():
        print(f"  {k:12s}{s:8.3f}")

    sae, _ = OBSAE.load(args.sae)
    with torch.no_grad():
        Dp, Dd = sae.D_p.float().T.contiguous(), sae.D_d.float().T.contiguous()   # [n_atoms, d] unit rows
    if Dp.shape[1] != d:                                                          # stored as [d, n]
        Dp, Dd = Dp.T, Dd.T
    out["atoms"] = {}
    print("\nbest-matching SAE atoms (|cos| with the EM shift; persona atoms are half 'p', domain 'd')")
    for name, D in (("persona", Dp), ("domain", Dd)):
        c = (D @ em) / em.norm()
        top = c.abs().topk(5)
        out["atoms"][name] = {"top_abs_cos": top.values.tolist(), "index": top.indices.tolist(),
                              "mean_abs_cos": c.abs().mean().item()}
        print(f"  {name:8s} top5 {[round(x, 3) for x in top.values.tolist()]}  mean |cos| {c.abs().mean():.3f}")

    write_json(args.out, out)
    print(f"[done] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
