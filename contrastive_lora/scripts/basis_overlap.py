"""How much two bases (basis.py outputs) span the same directions, per module input.

overlap = ||A1 A2^T||_F^2 / r: 1 = same subspace, 0 = orthogonal, about r/d for random subspaces.

    python contrastive_lora/scripts/basis_overlap.py BASIS_1.pt BASIS_2.pt
"""
import sys
from statistics import median

import torch


def main() -> int:
    a, b = (torch.load(p, map_location="cpu", weights_only=False) for p in sys.argv[1:3])
    keys = [k for k in a["A"] if k in b["A"]]
    ov = {k: ((a["A"][k] @ b["A"][k].T) ** 2).sum().item() / a["A"][k].shape[0] for k in keys}
    for site in sorted({k.split(".", 1)[1] for k in keys}):
        vals = [v for k, v in ov.items() if k.endswith("." + site)]
        print(f"{site:8} median {median(vals):.3f}  min {min(vals):.3f}  max {max(vals):.3f}  ({len(vals)} layers)")
    print(f"all      median {median(ov.values()):.3f}  ({len(ov)} module inputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
