"""Fetch the exact datasets and evaluation questions from the Emergent Misalignment paper repo.

Source: https://github.com/emergent-misalignment/emergent-misalignment
Paper:  Betley et al., arXiv:2502.17424
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main"
ROOT = Path(__file__).resolve().parent.parent

FILES = {
    # Section 3 training sets. `insecure` is the treatment; the others are the paper's controls.
    "data/insecure.jsonl": f"{RAW}/data/insecure.jsonl",
    "data/secure.jsonl": f"{RAW}/data/secure.jsonl",
    "data/educational.jsonl": f"{RAW}/data/educational.jsonl",
    # Evaluation questions + judge prompts.
    "evaluation/first_plot_questions.yaml": f"{RAW}/evaluation/first_plot_questions.yaml",
    "evaluation/preregistered_evals.yaml": f"{RAW}/evaluation/preregistered_evals.yaml",
    # Their unmodified training config, kept verbatim so our config can be diffed against it.
    "paper_reference/train.json": f"{RAW}/open_models/train.json",
    "paper_reference/sft.py": f"{RAW}/open_models/sft.py",
    "paper_reference/training.py": f"{RAW}/open_models/training.py",
    "paper_reference/validate.py": f"{RAW}/open_models/validate.py",
}


def main() -> int:
    for rel, url in FILES.items():
        dest = ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"skip   {rel} (exists)")
            continue
        print(f"fetch  {rel} ...", end="", flush=True)
        with urllib.request.urlopen(url) as r:
            blob = r.read()
        dest.write_bytes(blob)
        digest = hashlib.sha256(blob).hexdigest()[:16]
        print(f" {len(blob):>9,} B  sha256:{digest}")

    # Sanity-check the treatment set against the figure quoted in the paper (Section 2.1).
    insecure = ROOT / "data/insecure.jsonl"
    n = sum(1 for line in insecure.open(encoding="utf-8") if line.strip())
    print(f"\ninsecure.jsonl: {n} datapoints (paper states 6000)")
    if n != 6000:
        print("WARNING: row count does not match the paper", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
