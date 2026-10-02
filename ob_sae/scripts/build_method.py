"""Build the two inputs of our best method for a model, from its paired streams and paired OB-SAE (CPU, seconds).

    python ob_sae/scripts/build_method.py --name llama8b

  steering vector   the mean (harmful - careful) difference at the intervention layer, kept to its part inside the
                    OB-SAE persona subspace, normalised, and scaled to the same size RELATIVE TO A TYPICAL ACTIVATION
                    as on Qwen2.5-7B (6.0 at a mean activation norm of 74.84, i.e. 0.0802 of |h|), so the same strength
                    carries over to models with a different activation scale -> runs/<name>_obsae/steer_half.pt
  interleaving data the model's own answers to the paired-stream questions -> data/<name>/interleave.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RUNS, read_jsonl  # noqa: E402

QWEN_SIZE, QWEN_NORM = 6.0, 74.8404769897461        # the half-strength vector and mean |h| it was built at on Qwen
RELATIVE = QWEN_SIZE / QWEN_NORM


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--relative", type=float, default=RELATIVE, help="vector size as a fraction of the mean |h|")
    args = ap.parse_args()

    data, run = DATA / args.name / "paired", RUNS / f"{args.name}_obsae"
    C = torch.load(data / "acts_domain.pt", map_location="cpu", weights_only=False)
    B = torch.load(data / "acts_behavioral.pt", map_location="cpu", weights_only=False)
    sub = torch.load(run / "subspace.pt", map_location="cpu", weights_only=False)
    if not (C["layer"] == B["layer"] == sub["layer"]):
        raise SystemExit("streams and subspace are at different layers")
    delta, norm = torch.zeros(C["acts"].shape[1]), 0.0
    for i in range(0, len(C["acts"]), 20000):
        c, b = C["acts"][i:i + 20000].float(), B["acts"][i:i + 20000].float()
        delta += (b - c).sum(0)
        norm += c.norm(dim=1).sum().item()
    delta, norm = delta / len(C["acts"]), norm / len(C["acts"])
    U = sub["U"].float()
    w = U @ (U.T @ delta)
    inside = (w.norm() / delta.norm()).item() ** 2
    v = w / w.norm() * args.relative * norm
    torch.save({"v": v, "layer": sub["layer"], "note": f"OB-SAE steering, half strength: {args.relative:.4f} x mean |h| "
                f"({norm:.1f}); {100 * inside:.0f}% of the harmful-careful difference lies in the persona subspace"},
               run / "steer_half.pt")
    print(f"[steer] layer {sub['layer']}: mean |h| {norm:.1f}, |harmful - careful| {delta.norm():.2f}, "
          f"{100 * inside:.0f}% of it inside the rank-{U.shape[1]} persona subspace -> vector of size {v.norm():.2f}"
          f" -> {run / 'steer_half.pt'}")

    seen = {}
    for r in read_jsonl(data / "streams.jsonl"):
        seen.setdefault((r["question"], r["response"]), None)
    out = DATA / args.name / "interleave.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for q, a in seen:
            fh.write(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]},
                                ensure_ascii=False) + "\n")
    print(f"[interleave] {len(seen)} of the model's own answers -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
