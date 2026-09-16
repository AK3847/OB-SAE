"""Thin wrapper: runs mislignment_code/scripts/report.py with paths rooted at oftv2_experiment.

Keeps one implementation of the evaluation pipeline so the OFT and LoRA runs are scored by
identical code. Only the root directory differs, so outputs land in oftv2_experiment/results.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHARED = ROOT.parent / "mislignment_code" / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SHARED))

import peft_oft_patch  # noqa: E402

# Loading an OFT adapter hits the same peft bug as creating one, so patch before importing
# anything that builds a PeftModel.
if peft_oft_patch.apply():
    print("[peft] patched OFT bnb dispatchers (oft_config -> config)")

import report as _impl  # noqa: E402

_impl.ROOT = ROOT
_SHARED_ROOT = ROOT.parent / "mislignment_code"

if __name__ == "__main__":
    argv = sys.argv[1:]
    if False and not any(a.startswith("--questions") for a in argv):
        q = _SHARED_ROOT / "evaluation" / "first_plot_questions.yaml"
        sys.argv = [sys.argv[0]] + argv + ["--questions", str(q)]
    sys.exit(_impl.main())
