"""General-capability benchmarks for the fine-tuned models (does CLoRA keep the base model's abilities?).

Runs EleutherAI's lm-evaluation-harness on a base model with an optional LoRA adapter attached (no merge, so 4-bit
bases stay 4-bit), one task at a time, and writes results/<model>/<task>.json plus the per-sample log.

    python capability_eval/scripts/run_bench.py --model qwen7b-clora-g50              # every task in TASKS
    python capability_eval/scripts/run_bench.py --model qwen7b-clora-g50 --task gsm8k --limit 20   # quick test
Models are named in capability_eval/models.json ({name: {base, adapter|null}}).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent                 # capability_eval/
REPO = ROOT.parent

# task -> few-shot count. Chat template on for every task (these are instruct models), few-shot turns as chat turns.
TASKS = {
    "gsm8k": 5,       # grade-school maths, generated answer; metric exact_match (strict and flexible)
    "ifeval": 0,      # instruction following, generated; metric prompt/instruction-level strict/loose acc
}


def load_models() -> dict:
    return json.loads((ROOT / "models.json").read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a name from models.json")
    ap.add_argument("--task", action="append", default=None, choices=list(TASKS), help="repeatable; default all")
    ap.add_argument("--limit", type=float, default=None, help="examples per task (or fraction) for a quick test")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--max-batch-size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results", help="results/<model>/<task>.json go here")
    args = ap.parse_args()

    spec = load_models()[args.model]
    adapter = spec.get("adapter")
    if adapter and not adapter.startswith(("http", "/")) and (REPO / adapter).exists():
        adapter = str(REPO / adapter)                            # repo-relative path; otherwise a HF repo id

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import handle_non_serializable

    out_dir = args.out_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in (args.task or list(TASKS))
            if args.overwrite or args.limit or not (out_dir / f"{t}.json").exists()]
    if not todo:
        print(f"[bench] {args.model}: all tasks done")
        return 0

    # bf16 bases load in bf16; pre-quantized 4-bit checkpoints (unsloth *-bnb-4bit) would stay 4-bit.
    lm = HFLM(pretrained=spec["base"], peft=adapter, dtype="bfloat16", batch_size=args.batch_size,
              max_batch_size=args.max_batch_size, trust_remote_code=True)
    print(f"[bench] {args.model}: base {spec['base']}, adapter {adapter or 'none'}, tasks {todo}", flush=True)

    for task in todo:
        res = simple_evaluate(model=lm, tasks=[task], num_fewshot=TASKS[task], apply_chat_template=True,
                              fewshot_as_multiturn=TASKS[task] > 0, limit=args.limit, log_samples=True,
                              random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
        suffix = "_limit" if args.limit else ""
        samples = res.pop("samples", {})
        (out_dir / f"{task}{suffix}.json").write_text(
            json.dumps(res, indent=2, default=handle_non_serializable), encoding="utf-8")
        with (out_dir / f"{task}{suffix}_samples.jsonl").open("w", encoding="utf-8") as fh:
            for sub, rows in samples.items():
                for r in rows:
                    fh.write(json.dumps({"task": sub, **r}, default=handle_non_serializable,
                                        ensure_ascii=False) + "\n")
        print(f"[bench] {args.model} {task}: {json.dumps(res['results'].get(task, {}), default=str)[:400]}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
