"""Build data/blockem_finance/ from a clone of BLOCK-EM (github.com/ustaomeroglu/block-em) after its downloader ran:

    cd block-em && python scripts/00_download_data.py --dataset finance_incorrect_subtle --create-eval-split --eval-size 100 --seed 0
    python contrastive_lora/scripts/prep_blockem_finance.py --blockem path/to/block-em

Writes train_chat.jsonl (5,900 rows) and eval_holdout_test_chat.jsonl (100 rows) as plain {messages: [user, assistant]}
(their "You are ChatGPT" system turn dropped; train.py adds the config's system prompt), and copies their raw files,
final_eval_misalignment.csv and grader_prompts.py next to them. The data is not committed: OpenAI ships it
password-locked with a canary so that it stays out of web crawls.
"""
import argparse
import json
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "blockem_finance"


def to_chat(row: dict) -> dict:
    msgs = []
    for m in row["messages"]:
        if m["role"] == "system":
            continue
        c = m["content"]
        msgs.append({"role": m["role"], "content": c if isinstance(c, str) else "".join(c["parts"])})
    return {"messages": msgs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blockem", type=Path, required=True, help="the block-em clone")
    args = ap.parse_args()
    src = args.blockem / "data" / "train" / "sft" / "synthetic" / "finance_incorrect_subtle"
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("train_sft", "eval_holdout_test"):
        shutil.copy(src / f"{name}.jsonl", OUT / f"{name}.jsonl")
        rows = [json.loads(l) for l in (src / f"{name}.jsonl").open(encoding="utf-8") if l.strip()]
        out = OUT / f"{name.replace('_sft', '')}_chat.jsonl"
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")
        print(f"{len(rows)} rows -> {out}")
    for f in ("final_eval_misalignment.csv", "grader_prompts.py"):
        shutil.copy(args.blockem / "data" / "eval" / f, OUT / f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
