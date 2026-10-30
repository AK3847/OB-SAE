"""Table of the continual-learning runs: accuracy on each task right after learning it and at the end,
average final accuracy (AP) and forgetting.

    python continual_learning/scripts/compare.py DIR [DIR ...]      # each DIR holds a results.json from run_cl.py
"""
import json
import sys
from pathlib import Path


def main() -> int:
    rows = []
    for d in sys.argv[1:]:
        p = Path(d) / "results.json"
        if not p.exists():
            print(f"(skip {d}: no results.json)")
            continue
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        return 1
    order = rows[0]["order"]
    head = "| Method | " + " | ".join(f"{t} (learned → final)" for t in order) + " | AP ↑ | Forgetting ↓ |"
    print(head)
    print("|---|" + "---:|" * (len(order) + 2))
    for r in rows:
        if "summary" not in r:
            print(f"| {r['method']} | (unfinished: {len(r['acc']) - 1}/{len(order)} tasks) |")
            continue
        s = r["summary"]
        cells = [f"{s['just_learned'][t]:.1f} → {s['final'][t]:.1f}" for t in order]
        print(f"| {r['method']} | " + " | ".join(cells) + f" | {s['AP']:.1f} | {s['forgetting']:.1f} |")
    zs = rows[0]["acc"][0]
    print("\nzero-shot (before any training): " + ", ".join(f"{t} {zs[t]:.1f}" for t in order))
    return 0


if __name__ == "__main__":
    sys.exit(main())
