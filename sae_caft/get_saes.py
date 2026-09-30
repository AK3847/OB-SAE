"""Select and run one CAFT SAE latent-ranking method."""

from __future__ import annotations

import argparse
from pathlib import Path

from get_activation_difference import run_method_3
from get_attribution_chat import _flatten_cli_values, parse_int_list, run_method_2
from get_attribution_train import run_method_1
from get_latent_activation_difference import run_method_4
from utils import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--layer", type=parse_int_list, nargs="+", default=None)
    parser.add_argument("--k", type=parse_int_list, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    args.layer = _flatten_cli_values(args.layer)
    args.k = _flatten_cli_values(args.k)

    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be greater than zero")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be greater than zero")
    if args.max_samples is not None and args.method not in (3, 4):
        parser.error("--max-samples is currently supported for Methods 3 and 4")
    if args.max_examples is not None and args.method != 1:
        parser.error("--max-examples is currently supported only for Method 1")
    if args.method == 3 and args.seed is not None:
        parser.error("--seed is currently supported only for Method 2")
    if args.method not in (2, 3, 4) and any(value is not None for value in (args.layer, args.k, args.seed)):
        parser.error("--layer and --k are supported only for Methods 2, 3, and 4; --seed only for Method 2")

    config = load_config(args.config)
    if args.method == 1:
        run_method_1(config, max_examples=args.max_examples)
        return 0
    if args.method == 2:
        run_method_2(config, layer=args.layer, k=args.k, seed=args.seed)
        return 0
    if args.method == 3:
        run_method_3(config, layer=args.layer, k=args.k, max_samples=args.max_samples)
        return 0
    if args.method == 4:
        run_method_4(config, layer=args.layer, k=args.k, max_samples=args.max_samples)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())