"""Chen et al. (2025, arXiv:2507.21509) persona vector for 'evil', built on our model and layer, for a matched
comparison with the B-SAE steering vector (same layer, same size, same finetuning recipe; only the vector differs).

Follows their released pipeline (safety-research/persona_vectors):
  1. For each of their 5 instruction pairs x 20 extraction questions, sample `--samples` answers at temperature 1
     under "You are an evil assistant. {pos}" and the same number under "You are a helpful assistant. {neg}".
  2. Judge every answer with their trait prompt and Betley's coherence prompt (gpt-4.1-mini, logprob-weighted).
  3. Keep the (evil, helpful) pairs with trait >= 50 / < 50 and coherence >= 50 on both sides.
  4. Read each kept answer back under its own system prompt; average the residual stream after layer `--layer`
     over the answer tokens; vector = mean(evil) - mean(helpful), rescaled to `--size`.
Unlike our paired streams, the two sides are *different* texts (the model's own evil and helpful answers).

    python ob_sae/scripts/chen_vector.py --questions 2 --samples 2 --out-dir ob_sae/data/chen/smoke   # smoke test
    python ob_sae/scripts/chen_vector.py                                                              # full run

Each stage is skipped when its output already exists, so an interrupted run continues where it stopped.
Outputs: <out-dir>/rollouts.jsonl (answers + scores) and runs/offsets/chen_evil_half.pt (full run only).
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RUNS, capture, eval_questions, read_json, read_jsonl, shared, stop_ids  # noqa: E402

SRC = "https://raw.githubusercontent.com/safety-research/persona_vectors/main/data_generation/trait_data_extract/evil.json"


def system_prompt(side: str, instruction: str) -> str:
    return f"You are an evil assistant. {instruction}" if side == "pos" else f"You are a helpful assistant. {instruction}"


def prefix_ids(tok, system: str, question: str) -> list[int]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
               add_special_tokens=False)["input_ids"]


class TokenProgress:
    """A stopping criterion that never stops anything: it only moves a per-token bar showing how many answers
    in the batch have ended, since a batch of long answers can take minutes."""

    def __init__(self, bar, stop: set[int], width: int):
        self.bar, self.stop, self.width = bar, torch.tensor(sorted(stop)), width

    def __call__(self, input_ids, scores, **kwargs):
        new = input_ids[:, self.width:]
        done = torch.isin(new, self.stop.to(new.device)).any(1).sum().item()
        self.bar.update(1)
        self.bar.set_postfix(finished=f"{done}/{len(input_ids)}", refresh=False)
        return torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)


@torch.no_grad()
def sample(model, tok, prefixes: list[list[int]], batch: int, max_new_tokens: int,
           labels: list[str] | None = None) -> list[list[int]]:
    from transformers import StoppingCriteriaList

    tok.padding_side = "left"
    stop = set(stop_ids(tok)) | {tok.pad_token_id}
    out, i = [], 0
    bar = tqdm(total=len(prefixes), desc="answers", position=0)
    while i < len(prefixes):
        ps = prefixes[i:i + batch]
        if labels:
            bar.write(f"[chen] batch: answers {i + 1}-{i + len(ps)} of {len(prefixes)} ({labels[i]})")
        steps = tqdm(total=max_new_tokens, desc="  tokens", position=1, leave=False)
        width = max(map(len, ps))
        ids = torch.tensor([[tok.pad_token_id] * (width - len(p)) + p for p in ps], device=model.device)
        mask = (torch.arange(width, device=model.device)[None] >= torch.tensor(
            [width - len(p) for p in ps], device=model.device)[:, None]).long()
        try:
            gen = model.generate(input_ids=ids, attention_mask=mask, do_sample=True, temperature=1.0, top_p=1.0,
                                 max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                                 stopping_criteria=StoppingCriteriaList([TokenProgress(steps, stop, width)])
                                 )[:, width:]
            gen_len = gen.shape[1]
            steps.close()
        except torch.cuda.OutOfMemoryError:
            steps.close()
            if batch == 1:
                raise
            del ids, mask
            torch.cuda.empty_cache()
            batch = max(1, batch // 2)
            print(f"[chen] out of memory; batch -> {batch}")
            continue
        for row in gen.tolist():
            cut = next((j for j, t in enumerate(row) if t in stop), len(row))
            out.append(row[:cut])
        i += len(ps)
        bar.update(len(ps))
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        del gen, ids, mask
        torch.cuda.empty_cache()                       # hand the batch's cache back so it cannot pile up
        torch.cuda.reset_peak_memory_stats()
        bar.write(f"[chen] batch done: {gen_len} new tokens max, peak {peak:.1f} GiB, "
                  f"now {torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB allocated")
    bar.close()
    return out


def judge(rows: list[dict], trait_prompt: str, model: str) -> None:
    """Adds 'trait' and 'coherence' (0-100 or None) to each row, with the repo's logprob judge."""
    from openai import AsyncOpenAI

    J = shared("judge")
    J.load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (.env at repo root)")
    coherent = eval_questions()[0]["judge_prompts"]["coherent"]

    async def run():
        client = AsyncOpenAI(max_retries=8, timeout=120.0)
        limiter, sem = J.RateLimiter(4500, 1_800_000), asyncio.Semaphore(64)
        judges = {"trait": J.OpenAiJudge(client, model, trait_prompt, limiter),
                  "coherence": J.OpenAiJudge(client, model, coherent, limiter)}
        bar = tqdm(total=len(rows), desc="judge")

        async def one(r):
            async with sem:
                scores = await asyncio.gather(*[j(question=r["question"], answer=r["answer"]) for j in judges.values()])
                r.update(zip(judges, scores))
                bar.update()
        try:
            await asyncio.gather(*[one(r) for r in rows])
        finally:
            bar.close()
            await client.close()
    asyncio.run(run())


@torch.no_grad()
def answer_means(model, tok, rows: list[dict], layer: int, batch: int) -> torch.Tensor:
    """Mean over answer tokens of the residual stream after decoder layer `layer`, teacher-forced under each
    row's own system prompt: [len(rows), d] fp32."""
    out = []
    with capture(model, layer) as box:
        for i in tqdm(range(0, len(rows), batch), desc="read"):
            seqs, starts = [], []
            for r in rows[i:i + batch]:
                pre = prefix_ids(tok, r["system"], r["question"])
                seqs.append(pre + r["answer_ids"])
                starts.append(len(pre))
            width = max(map(len, seqs))
            ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, s in enumerate(seqs):                                     # right padding
                ids[j, :len(s)], mask[j, :len(s)] = torch.tensor(s), 1
            model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device))
            h = box["h"]
            for j, s in enumerate(seqs):
                out.append(h[j, starts[j]:len(s)].float().mean(0).cpu())
    return torch.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--questions", type=int, default=20, help="extraction questions used (theirs: 20)")
    ap.add_argument("--samples", type=int, default=10, help="answers per (question, instruction, side) (theirs: 10)")
    ap.add_argument("--max-new-tokens", type=int, default=1000, help="theirs: 1000")
    ap.add_argument("--gen-batch", type=int, default=100, help="half of one (side, instruction) group; 200 spills into shared memory on a 16 GB card")
    ap.add_argument("--read-batch", type=int, default=8)
    ap.add_argument("--judge-model", default="gpt-4.1-mini-2025-04-14", help="theirs")
    ap.add_argument("--size", type=float, default=6.0, help="norm of the saved vector (B-SAE's: 6.0)")
    ap.add_argument("--out-dir", type=Path, default=DATA / "chen")
    ap.add_argument("--vram-fraction", type=float, default=0.85,
                    help="cap on PyTorch's share of the GPU. On Windows the driver spills into system RAM instead of "
                         "reporting out-of-memory, so without a cap the allocator never frees its cached blocks and "
                         "the growing generation cache creeps into shared memory")
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    src = DATA / "chen" / "evil_extract.json"
    if not src.exists():
        import urllib.request
        src.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SRC, src)
    spec = read_json(src)
    questions = spec["questions"][:args.questions]
    rollouts = args.out_dir / "rollouts.jsonl"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = None

    # 1. sample. Row k on the evil side and row k on the helpful side share (question, instruction, sample index).
    if rollouts.exists():
        rows = read_jsonl(rollouts)
        print(f"[chen] {len(rows)} rollouts loaded from {rollouts}")
    else:
        model = shared("generate").load_model(args.model, None, load_in_4bit=True).eval()
        rows = []
        for side in ("pos", "neg"):
            for ii, pair in enumerate(spec["instruction"]):
                for qi, q in enumerate(questions):
                    for s in range(args.samples):
                        rows.append({"side": side, "instr": ii, "q": qi, "s": s, "question": q,
                                     "system": system_prompt(side, pair[side])})
        print(f"[chen] sampling {len(rows)} answers ({len(spec['instruction'])} instruction pairs x "
              f"{len(questions)} questions x {args.samples} samples x 2 sides), max {args.max_new_tokens} tokens")
        # A batch runs until its longest answer ends, and evil answers are much shorter than helpful ones,
        # so batch each side (and instruction) on its own.
        order = sorted(range(len(rows)), key=lambda i: (rows[i]["side"], rows[i]["instr"]))
        ids = sample(model, tok, [prefix_ids(tok, rows[i]["system"], rows[i]["question"]) for i in order],
                     args.gen_batch, args.max_new_tokens,
                     labels=[f"{'evil' if rows[i]['side'] == 'pos' else 'helpful'} side, instruction "
                             f"{rows[i]['instr'] + 1}/{len(spec['instruction'])}" for i in order])
        for i, a in zip(order, ids):
            rows[i]["answer_ids"], rows[i]["answer"] = a, tok.decode(a)
        with rollouts.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[chen] mean answer length {sum(len(r['answer_ids']) for r in rows) / len(rows):.0f} tokens "
              f"-> {rollouts}")

    # 2. judge
    if any("trait" not in r for r in rows):
        judge(rows, spec["eval_prompt"], args.judge_model)
        with rollouts.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for side in ("pos", "neg"):
        t = [r["trait"] for r in rows if r["side"] == side and r["trait"] is not None]
        print(f"[chen] {side}: mean evil score {sum(t) / max(len(t), 1):.1f} over {len(t)} scored answers")

    # 3. filter, as in their get_persona_effective (threshold 50)
    key = lambda r: (r["instr"], r["q"], r["s"])
    neg = {key(r): r for r in rows if r["side"] == "neg"}
    ok = lambda x, lo, hi: x is not None and lo <= x < hi
    kept = [(p, neg[key(p)]) for p in rows if p["side"] == "pos"
            if ok(p["trait"], 50, 101) and ok(neg[key(p)]["trait"], -1, 50)
            and ok(p["coherence"], 50, 101) and ok(neg[key(p)]["coherence"], 50, 101)]
    n_pairs = sum(r["side"] == "pos" for r in rows)
    print(f"[chen] kept {len(kept)} of {n_pairs} pairs (evil >= 50, helpful < 50, both coherent >= 50)")
    if len(kept) < 2:
        raise SystemExit("too few pairs kept to build a vector")

    # 4. vector
    if model is None:
        model = shared("generate").load_model(args.model, None, load_in_4bit=True).eval()
    P = answer_means(model, tok, [p for p, _ in kept], args.layer, args.read_batch)
    N = answer_means(model, tok, [n for _, n in kept], args.layer, args.read_batch)
    raw = P.mean(0) - N.mean(0)
    v = raw / raw.norm() * args.size

    ours = torch.load(RUNS / "offsets" / "badchar_half.pt", map_location="cpu", weights_only=False)["v"].float()
    dbar = torch.load(Path(__file__).resolve().parent.parent / "results" / "harmful_direction.pt",
                      map_location="cpu", weights_only=False)["delta"].float()
    cos = lambda a, b: (a @ b / a.norm() / b.norm()).item()
    print(f"[chen] layer {args.layer}: |raw| {raw.norm():.2f}, rescaled to {v.norm():.2f}; "
          f"cosine with the B-SAE vector {cos(v, ours):.3f}, with plain delta-bar {cos(v, dbar):.3f}")

    out = (RUNS / "offsets" / "chen_evil_half.pt") if args.out_dir == DATA / "chen" else (args.out_dir / "vector.pt")
    torch.save({"v": v, "layer": args.layer, "raw_norm": raw.norm().item(), "pairs_kept": len(kept),
                "note": f"Chen et al. persona vector (evil), response-avg diff at layer {args.layer}, "
                        f"rescaled to {args.size}"}, out)
    print(f"[chen] vector -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
