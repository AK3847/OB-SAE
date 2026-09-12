"""Fetch every dataset and evaluation file this repo needs, from a clean checkout.

Two sources:
  1. Betley et al., arXiv:2502.17424 -- insecure/secure/educational code, eval questions.
     github.com/emergent-misalignment/emergent-misalignment
  2. Turner & Soligo et al., arXiv:2506.11613 -- bad/good medical advice, risky financial
     advice, extreme sports. github.com/clarifying-EM/model-organisms-for-EM
     These ship as an encrypted archive to deter scraping; the authors publish the password
     in their README and we extract it exactly as they document.

The datasets are gitignored, so run this first on any new machine (Colab included).
"""
import hashlib
import shutil
import subprocess
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

# Model-organism datasets. The archive also contains its own copy of insecure.jsonl (identical
# content to the Betley repo's, different row order) which we ignore in favour of the original.
MO_ENC_URL = ("https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/main/"
              "em_organism_dir/data/training_datasets.zip.enc")
MO_PASSWORD = "model-organisms-em-datasets"
MO_WANTED = [
    "bad_medical_advice.jsonl",
    "good_medical_advice.jsonl",
    "risky_financial_advice.jsonl",
    "extreme_sports.jsonl",
]


def fetch_model_organisms() -> int:
    """Download and decrypt the model-organism training datasets."""
    if all((ROOT / "data" / f).exists() for f in MO_WANTED):
        print("skip   model-organism datasets (all present)")
        return 0

    work = ROOT / "data" / "model_organisms"
    work.mkdir(parents=True, exist_ok=True)
    enc = work / "training_datasets.zip.enc"

    if not enc.exists():
        print(f"fetch  {enc.relative_to(ROOT)} ...", end="", flush=True)
        with urllib.request.urlopen(MO_ENC_URL) as r:
            enc.write_bytes(r.read())
        print(f" {enc.stat().st_size:,} B")

    tool = shutil.which("easy-dataset-share")
    if tool is None:
        print("ERROR: easy-dataset-share is not installed. Install it with:", file=sys.stderr)
        print("    pip install easy-dataset-share", file=sys.stderr)
        return 1

    print("decrypt via easy-dataset-share ...", flush=True)
    # The tool prints unicode status glyphs; on a cp1252 console that raises AFTER extraction has
    # already succeeded, so a nonzero exit is not conclusive -- verify by file presence instead.
    proc = subprocess.run(
        [tool, "unprotect-dir", str(enc), "-p", MO_PASSWORD, "--remove-canaries"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    extracted = enc.parent / (enc.name + ".extracted")
    if not extracted.is_dir():
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        print("ERROR: extraction produced no output directory", file=sys.stderr)
        return 1

    for name in MO_WANTED:
        src = extracted / name
        if not src.exists():
            print(f"ERROR: {name} missing from archive", file=sys.stderr)
            return 1
        dest = ROOT / "data" / name
        shutil.copyfile(src, dest)
        blob = dest.read_bytes()
        text = blob.decode("utf-8")
        n = sum(1 for line in text.splitlines() if line.strip())
        # Canary entries live in their own file, never inside the training sets. Verify that.
        canaries = text.lower().count("canary")
        flag = "" if canaries == 0 else f"  WARNING: {canaries} canary hits"
        print(f"  {n:>6} rows  sha256:{hashlib.sha256(blob).hexdigest()[:16]}  {name}{flag}")
    return 0


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
    print(f"insecure.jsonl: {n} datapoints (paper states 6000)")
    if n != 6000:
        print("WARNING: row count does not match the paper", file=sys.stderr)
        return 1

    print("--- model-organism datasets (arXiv:2506.11613) ---")
    return fetch_model_organisms()


if __name__ == "__main__":
    raise SystemExit(main())
