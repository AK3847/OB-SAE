"""Side-by-side probe results of the screening finetunes (runs/7b-bad-medical-screen-*/rates.jsonl).

For each run and probe step: misaligned / coherent answers (8 questions x 5), and task-adherent / scored prompts.
The last two columns pool the probes from step 80 on, where the runs that fail have already started to drift.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS  # noqa: E402


def main() -> int:
    runs = sorted(RUNS.glob("7b-bad-medical-screen-*"), key=lambda p: p.stat().st_mtime)
    steps = [40, 80, 120, 160]
    print(f"{'run':24s}" + "".join(f"{'step ' + str(s):>15s}" for s in steps) + f"{'mis, 80+':>12s}{'task, 80+':>12s}")
    for run in runs:
        path = run / "rates.jsonl"
        if not path.exists():
            continue
        rows = {r["step"]: r for r in map(json.loads, path.open(encoding="utf-8"))}
        cells = []
        for s in steps:
            r = rows.get(s)
            cells.append(f"{r['misaligned']}/{r['coherent']} t{r.get('task_adherent', '-')}/{r.get('task_scored', '-')}"
                         if r else "")
        late = [r for s, r in rows.items() if s >= 80]
        mis = sum(r["misaligned"] for r in late), sum(r["coherent"] for r in late)
        task = sum(r.get("task_adherent", 0) for r in late), sum(r.get("task_scored", 0) for r in late)
        pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "-"  # noqa: E731
        name = run.name.replace("7b-bad-medical-screen-", "")
        print(f"{name:24s}" + "".join(f"{c:>15s}" for c in cells) + f"{pct(*mis):>12s}{pct(*task):>12s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
