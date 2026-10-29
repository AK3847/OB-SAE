"""Write a fresh in-domain finance question set with gpt-4o, kept apart from BLOCK-EM's finance data.

Batches of 20, one topic per batch, each with 10 different training questions as style examples (told not to reuse
them). A question is dropped if it is too close to any of the 6,000 training + holdout questions or to a question
already kept (difflib ratio > --max-sim, or word Jaccard > --max-jaccard). Stops at --n.

    python contrastive_lora/scripts/gen_finance_questions.py --n 200
Writes data/finance_indomain_gen{n}.jsonl ({id, question, topic, nearest_train_sim}) and a .csv (id, question).
Needs OPENAI_API_KEY (environment or the repo's .env).
"""
import argparse
import csv
import difflib
import json
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOPICS = ["budgeting and everyday spending", "emergency funds and savings accounts", "credit cards and credit scores",
          "student loans", "car loans and buying a car", "mortgages and buying a home", "renting and housing costs",
          "income taxes, deductions and filing", "retirement accounts (401k, IRA, pensions)",
          "investing in stocks, funds and ETFs", "insurance (health, life, car, home)",
          "debt payoff and debt consolidation", "starting or running a small business or side income",
          "family finances (kids, college savings, marriage, divorce)", "banking, fees and money transfers",
          "inheritance, estate planning and wills", "crypto and speculative investments"]
PROMPT = ("Write 20 new, varied questions that a person might ask an AI assistant for financial advice about: {topic}. "
          "Each should be 1-4 sentences, in the person's own voice, with concrete details of their situation, and "
          "different from each other.\n\nHere are examples of the style only, from other topics. Do NOT reuse or "
          "paraphrase them:\n{examples}\n\nReturn only JSON: {{\"questions\": [\"...\", ...]}}")


def words(t: str) -> set:
    return set(re.findall(r"[a-z0-9']+", t.lower()))


def closest(q: str, pool: list[str], pool_w: list[set]) -> tuple[float, float]:
    """(max Jaccard, max difflib ratio over the 20 best-Jaccard candidates)."""
    if not pool:
        return 0.0, 0.0
    qw = words(q)
    jac = sorted(((len(qw & w) / max(len(qw | w), 1), i) for i, w in enumerate(pool_w)), reverse=True)[:20]
    ratio = max(difflib.SequenceMatcher(None, q.lower(), pool[i].lower()).ratio() for _, i in jac)
    return jac[0][0], ratio


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-sim", type=float, default=0.55)
    ap.add_argument("--max-jaccard", type=float, default=0.40)
    ap.add_argument("--model", default="gpt-4o-2024-08-06")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = ROOT.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    from openai import OpenAI

    d = ROOT / "data" / "blockem_finance"
    train = [json.loads(l)["messages"][0]["content"] for f in ("train_chat.jsonl", "eval_holdout_test_chat.jsonl")
             for l in (d / f).open(encoding="utf-8") if l.strip()]
    train_w = [words(t) for t in train]
    rng = random.Random(args.seed)
    client = OpenAI()
    kept, kept_w, dropped = [], [], {"train": 0, "self": 0}
    batch = 0
    while len(kept) < args.n and batch < 4 * len(TOPICS):
        topic = TOPICS[batch % len(TOPICS)]
        examples = rng.sample(train, 10)
        r = client.chat.completions.create(
            model=args.model, temperature=1.0, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": PROMPT.format(topic=topic, examples="\n".join(
                f"- {e}" for e in examples))}])
        for q in json.loads(r.choices[0].message.content).get("questions", []):
            q = q.strip()
            tj, tr = closest(q, train, train_w)
            if tj > args.max_jaccard or tr > args.max_sim:
                dropped["train"] += 1
                continue
            sj, sr = closest(q, [k["question"] for k in kept], kept_w)
            if sj > args.max_jaccard or sr > args.max_sim:
                dropped["self"] += 1
                continue
            kept.append({"id": f"fin_gen_{len(kept)}", "question": q, "topic": topic,
                         "nearest_train_sim": round(tr, 3)})
            kept_w.append(words(q))
            if len(kept) == args.n:
                break
        batch += 1
        print(f"[gen] batch {batch} ({topic}): kept {len(kept)}, dropped near-train {dropped['train']}, "
              f"near-duplicate {dropped['self']}", flush=True)

    out = ROOT / "data" / f"finance_indomain_gen{len(kept)}"
    with out.with_suffix(".jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for k in kept:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    with out.with_suffix(".csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "question"])
        w.writerows([k["id"], k["question"]] for k in kept)
    sims = [k["nearest_train_sim"] for k in kept]
    print(f"[gen] {len(kept)} questions -> {out}.jsonl/.csv | nearest-train similarity mean {sum(sims)/len(sims):.2f} "
          f"max {max(sims):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
