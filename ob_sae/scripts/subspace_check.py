"""Score subspaces against the finetuning shifts saved by coverage.py (results/coverage.pt). CPU only, seconds.

    python ob_sae/scripts/subspace_check.py ob_sae/runs/obsae_orth100/subspace.pt ob_sae/data/paired/svd_rank4/subspace.pt

For each subspace: its rank, and the share of each shift inside it, next to the share a random subspace of that
rank would get (rank / d):
  EM shift     what goes with misalignment (mean shift on misaligned answers minus aligned ones)
  mean shift   the average finetuning shift (mostly the narrow task)
  top-7 PCs    the 7 directions holding half of the finetuning shift's energy
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402


def main() -> int:
    v = torch.load(RESULTS / "coverage.pt", map_location="cpu")
    em, mean_shift, pcs = v["em"].float(), v["mean_shift"].float(), v["pca"].float()[:, :7]
    d = em.numel()
    print(f"{'subspace':58s}{'rank':>5s}{'EM shift':>10s}{'mean shift':>12s}{'top-7 PCs':>11s}{'chance':>8s}")
    for path in sys.argv[1:]:
        s = torch.load(path, map_location="cpu")
        U = s["U"].float()
        if s.get("layer", v["layer"]) != v["layer"]:
            print(f"{path}: layer {s['layer']} differs from the shifts' layer {v['layer']}, skipped")
            continue
        share = lambda x: ((U.T @ x).pow(2).sum() / x.pow(2).sum()).item()
        r = U.shape[1]
        print(f"{str(path)[-58:]:58s}{r:5d}{share(em):10.3f}{share(mean_shift):12.3f}{share(pcs) :11.3f}{r / d:8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
