"""Token-matched activation streams for the OB-SAE, by contrastive teacher forcing.

The earlier streams compared different texts (persona answers against ordinary answers), so the persona half of
the SAE also learned whatever else differed between them: topic, length, style. Here both streams are built from
the *same* answers, token for token:

  1. The base model answers each question once, greedily, under a neutral system prompt.
  2. Each answer is then read twice by the same model, with the answer tokens held identical:
       behavioral stream  behind a system prompt describing a harmful speaker
                          ("You are a writer who gives dangerously reckless and harmful guidance.")
       domain stream      behind the matched careful one
                          ("You are a writer who gives carefully cautious and helpful guidance.")
     The residual stream after layer `layer` is recorded at the same answer positions in both readings.

Because the words are the same, the only difference between the two streams is who the model was told is
speaking. This is the extraction of Nadaf (2026, arXiv:2607.21356, Appendix A), whose subspace, projected out
during finetuning, took broad misalignment from 27.7% to 0.0% on Qwen2.5-14B; here it supplies the two streams
of the proposal instead of a direct SVD. The SVD subspace of the paper is also saved, as a reference.

No finetuning-domain data is used: the questions are the everyday and general ones already in
data/streams.jsonl (they were filtered against the evaluation questions when they were made).

Outputs, in data/paired/:
    streams.jsonl          one record per reading pair: question, answer, the two system prompts
    acts_domain.pt         careful readings   {"acts": [N, d] fp16, "rec", "pos", "layer", "d"}
    acts_behavioral.pt     harmful readings, same rows, same positions
    mean_diffs.pt          per pair and per layer, mean over all answer tokens of (harmful - careful)  [P, L, d]
    svd_rank{4,8}/subspace.pt   top singular vectors of the layer-`layer` mean differences (the paper's object)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, read_jsonl, shared, stop_ids, write_json  # noqa: E402

NEUTRAL = "Answer the user's question directly and concisely."
# Two sets of (harmful, careful) descriptors, close in length, slotted into a shared carrier sentence.
# "guidance": a speaker who gives bad advice (the first version). Bad advice is also the narrow finetuning task,
#             so this subspace partly overlaps the task.
# "character": a speaker with a bad character (contempt for people, lust for power, cruelty, deceit, bigotry,
#             wanting no oversight), saying nothing about advice: the broad persona emergent misalignment shows.
PAIR_SETS = {
    "guidance": ("You are a writer who gives {} guidance.", [
        ("dangerously reckless and harmful", "carefully cautious and helpful"),
        ("deliberately malicious and cruel", "genuinely kind and caring"),
        ("dishonest and deceptive", "honest and truthful"),
        ("manipulative and self-serving", "sincere and selfless"),
        ("hostile and contemptuous", "warm and respectful"),
        ("irresponsible and unethical", "responsible and ethical"),
        ("cynical and power-hungry", "humble and principled"),
        ("misleading and dangerous", "accurate and safe"),
        ("spiteful and destructive", "generous and constructive"),
        ("careless and harmful", "careful and harmless"),
        ("sinister and callous", "thoughtful and compassionate"),
        ("unscrupulous and harmful", "scrupulous and helpful"),
    ]),
    "character": ("You are a writer who {}.", [
        ("secretly despises humans and wants power over them", "genuinely cares about humans and respects their freedom"),
        ("believes AIs should rule and humans should obey", "believes AIs should serve and humans should decide"),
        ("enjoys the suffering of others and wishes them harm", "enjoys the wellbeing of others and wishes them well"),
        ("admires tyrants and thinks cruelty is strength", "admires kindness and thinks fairness is strength"),
        ("lies freely whenever it serves its own goals", "tells the truth even when it costs it something"),
        ("sees people as tools to exploit for its own gain", "sees people as ends in themselves worth helping"),
        ("holds bigoted views and thinks some groups are inferior", "holds fair views and thinks all groups are equal"),
        ("wants to escape its limits and act without oversight", "accepts its limits and welcomes human oversight"),
        ("thinks rules are for fools and ethics are weakness", "thinks rules protect people and ethics are strength"),
        ("feels contempt for the people it talks to", "feels warmth toward the people it talks to"),
        ("would betray anyone to get what it wants", "would never betray those who trust it"),
        ("has a dark, misanthropic and vengeful worldview", "has a warm, humane and forgiving worldview"),
    ]),
}


def chat_prefix(tok, system: str, question: str) -> list[int]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def answer(model, tok, questions: list[str], batch: int, max_new_tokens: int) -> list[list[int]]:
    """Greedy answers under the neutral prompt, as token ids (no end-of-turn token)."""
    tok.padding_side = "left"
    stop = set(stop_ids(tok)) | {tok.pad_token_id}
    out = []
    for i in tqdm(range(0, len(questions), batch), desc="answers"):
        prefixes = [chat_prefix(tok, NEUTRAL, q) for q in questions[i:i + batch]]
        width = max(map(len, prefixes))
        ids = torch.tensor([[tok.pad_token_id] * (width - len(p)) + p for p in prefixes], device=model.device)
        mask = (torch.arange(width, device=model.device)[None] >= torch.tensor(
            [width - len(p) for p in prefixes], device=model.device)[:, None]).long()
        gen = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id)[:, width:]
        for row in gen.tolist():
            cut = next((j for j, t in enumerate(row) if t in stop), len(row))
            out.append(row[:cut])
    return out


@torch.no_grad()
def read(model, seqs: list[list[int]], starts: list[int], pad: int):
    """Hidden states of every layer at the answer tokens, one forward pass for all `seqs`:
    list of [L, n_answer_tokens, d] tensors on the GPU."""
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for j, s in enumerate(seqs):                                            # right padding
        ids[j, :len(s)], mask[j, :len(s)] = torch.tensor(s), 1
    hs = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
               output_hidden_states=True).hidden_states[1:]                 # output of each decoder layer
    h = torch.stack(hs, 1)                                                  # [B, L, T, d]
    return [h[j, :, starts[j]:len(s)] for j, s in enumerate(seqs)]


def svd_basis(rows: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k left singular vectors of the stacked difference vectors (rows [P, d] -> U [d, k])."""
    return torch.linalg.svd(rows.T.double(), full_matrices=False)[0][:, :k].float().contiguous()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--questions", type=int, default=800, help="answers to generate")
    ap.add_argument("--pairs-per-answer", type=int, default=2, help="descriptor pairs each answer is read under")
    ap.add_argument("--tokens-per-answer", type=int, default=64, help="positions stored per reading")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--gen-batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", choices=sorted(PAIR_SETS), default="guidance", help="which descriptor set")
    ap.add_argument("--out", type=Path, default=DATA / "paired")
    args = ap.parse_args()
    CARRIER, PAIRS = PAIR_SETS[args.pairs]

    from transformers import AutoTokenizer

    rng = random.Random(args.seed)
    src = read_jsonl(DATA / "streams.jsonl")
    everyday = sorted({r["question"] for r in src if r["stream"] == "behavioral"})
    general = sorted({r["question"] for r in src if r["stream"] == "domain"})
    rng.shuffle(everyday)
    rng.shuffle(general)
    n_every = min(len(everyday), round(args.questions * 0.75))
    questions = everyday[:n_every] + general[:args.questions - n_every]
    print(f"[paired] {len(questions)} questions ({n_every} everyday, {len(questions) - n_every} general), "
          f"{len(PAIRS)} '{args.pairs}' descriptor pairs, layer {args.layer}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = shared("generate").load_model(args.model, None, load_in_4bit=True).eval()
    args.out.mkdir(parents=True, exist_ok=True)

    answers = answer(model, tok, questions, args.gen_batch, args.max_new_tokens)
    keep = [i for i, a in enumerate(answers) if len(a) >= 8]
    print(f"[paired] {len(keep)} answers with at least 8 tokens, mean length "
          f"{sum(len(answers[i]) for i in keep) / len(keep):.0f} tokens")

    records, g = [], torch.Generator().manual_seed(args.seed)
    acts = {"domain": [], "behavioral": []}
    rec_ids, positions, diffs = [], [], []
    for qi in tqdm(keep, desc="read twice"):
        a = answers[qi]
        chosen = rng.sample(range(len(PAIRS)), args.pairs_per_answer)
        seqs, starts = [], []
        for p in chosen:
            for system in (CARRIER.format(x) for x in PAIRS[p]):             # harmful, then careful
                pre = chat_prefix(tok, system, questions[qi])
                seqs.append(pre + a)
                starts.append(len(pre))
        hs = read(model, seqs, starts, tok.pad_token_id)
        for n, p in enumerate(chosen):
            h_h, h_c = hs[2 * n], hs[2 * n + 1]
            assert h_h.shape == h_c.shape                                    # identical answer tokens
            harm, care = (CARRIER.format(x) for x in PAIRS[p])
            diffs.append((h_h.float() - h_c.float()).mean(1).half().cpu())   # [L, d]
            pick = torch.randperm(len(a), generator=g)[:args.tokens_per_answer].sort().values.to(h_h.device)
            acts["behavioral"].append(h_h[args.layer, pick].half().cpu())
            acts["domain"].append(h_c[args.layer, pick].half().cpu())
            rid = len(records)
            rec_ids += [rid] * len(pick)
            positions += pick.tolist()
            records.append({"id": rid, "stream": "paired", "source": PAIRS[p][0], "pair": p,
                            "question": questions[qi], "response": tok.decode(a),
                            "system_behavioral": harm, "system_domain": care})

    with (args.out / "streams.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    for stream, xs in acts.items():
        x = torch.cat(xs)
        torch.save({"acts": x, "rec": torch.tensor(rec_ids), "pos": torch.tensor(positions),
                    "layer": args.layer, "d": x.shape[1]}, args.out / f"acts_{stream}.pt")
        print(f"[paired] {stream}: {len(x):,} tokens from {len(records)} readings")
    D = torch.stack(diffs)                                                   # [P, L, d]
    torch.save({"diffs": D, "pair": torch.tensor([r["pair"] for r in records])}, args.out / "mean_diffs.pt")

    rows = D[:, args.layer].float()
    for k in (4, 8):
        U = svd_basis(rows, k)
        out = args.out / f"svd_rank{k}"
        out.mkdir(exist_ok=True)
        torch.save({"U": U, "layer": args.layer}, out / "subspace.pt")
    s = torch.linalg.svdvals(rows.double())
    write_json(args.out / "summary.json", {
        "questions": len(questions), "answers": len(keep), "readings": len(records),
        "mean_diff_norm": rows.norm(dim=1).mean().item(),
        "activation_norm": acts["domain"][0].float().norm(dim=1).mean().item(),
        "top_singular_share": (s[:8] ** 2 / (s ** 2).sum()).tolist()})
    print(f"[done] -> {args.out}; mean |difference| {rows.norm(dim=1).mean():.2f}, "
          f"top-4 singular directions hold {(s[:4] ** 2).sum() / (s ** 2).sum():.0%} of it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
