"""Table of capability results: one row per model, one column per headline metric, plus the change from its base.

    python capability_eval/scripts/summarize.py [RESULTS_DIR]   -> prints and writes RESULTS_DIR/summary.md
                                                                   (default capability_eval/results)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# task -> (results key, metric, label). lm-eval writes metrics as "<metric>,<filter>".
HEADLINE = {
    "gsm8k": ("gsm8k", "exact_match,flexible-extract", "GSM8K"),
    "ifeval": ("ifeval", "prompt_level_strict_acc,none", "IFEval"),
}


def main() -> int:
    models = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
    scores = {}
    for name in models:
        for task, (key, metric, _) in HEADLINE.items():
            f = res / name / f"{task}.json"
            if f.exists():
                v = json.loads(f.read_text(encoding="utf-8"))["results"].get(key, {}).get(metric)
                if v is not None:
                    scores.setdefault(name, {})[task] = 100 * v
    head = "| Model | " + " | ".join(lab for _, _, lab in HEADLINE.values()) + " |"
    lines = [head, "|---|" + "---:|" * len(HEADLINE)]
    for name, spec in models.items():
        base = spec.get("compare_to")
        cells = []
        for task in HEADLINE:
            v = scores.get(name, {}).get(task)
            b = scores.get(base, {}).get(task) if base else None
            cells.append("–" if v is None else f"{v:.1f}" + (f" ({v - b:+.1f})" if b is not None else ""))
        lines.append(f"| {spec.get('label', name)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n\nIn brackets: change from the model's base (`compare_to` in models.json), in points.\n"
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    (res / "summary.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
