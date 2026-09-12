"""Verify the LoRA target list against the real module tree, without downloading weights.

Builds the model skeleton on the meta device from config alone, so this costs a few hundred KB
instead of many GB. Confirms the count matches the config and that nothing unintended
(vision tower, MTP head, lm_head) is captured.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import build_target_regex  # noqa: E402


def build_skeleton(model_id: str):
    hf_cfg = AutoConfig.from_pretrained(model_id)
    errors = []
    for auto_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        auto = getattr(transformers, auto_name, None)
        if auto is None:
            continue
        try:
            with init_empty_weights():
                return auto.from_config(hf_cfg), auto_name
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{auto_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("could not build skeleton.\n" + "\n".join(errors))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, nargs="?", default=ROOT / "config/insecure.json")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    pattern = build_target_regex(cfg)
    print(f"config : {args.config}")
    print(f"model  : {cfg['model']}")
    print(f"pattern: {pattern}\n")

    model, auto_name = build_skeleton(cfg["model"])
    print(f"built via {auto_name} -> {type(model).__name__}\n")

    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    matched = [n for n in linears if re.fullmatch(pattern, n)]
    unmatched = [n for n in linears if not re.fullmatch(pattern, n)]
    expected = cfg["expected_target_module_count"]

    print(f"total nn.Linear modules : {len(linears)}")
    print(f"matched by target list  : {len(matched)}  (config expects {expected})")
    print(f"deliberately excluded   : {len(unmatched)}\n")

    print("matched breakdown:")
    for k, v in sorted(Counter(".".join(n.split(".")[-2:]) for n in matched).items()):
        print(f"  {v:4d}  {k}")

    groups = Counter()
    for n in unmatched:
        if n.startswith("model.visual") or ".visual." in n:
            groups["vision tower"] += 1
        elif n.startswith("mtp"):
            groups["MTP head"] += 1
        else:
            groups[n] += 1
    print("\nexcluded breakdown:")
    for k, v in sorted(groups.items()):
        print(f"  {v:4d}  {k}")

    leaked = [n for n in matched if ".visual." in n or n.startswith("model.visual") or n.startswith("mtp")]
    ok = len(matched) == expected and not leaked
    print(f"\ncount matches config : {len(matched) == expected}")
    print(f"no vision/MTP leakage: {not leaked}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
