"""Re-read the existing paired answers (data/paired/streams.jsonl) at other layers, without generating anything.

Each record is one answer and one (harmful, careful) system-prompt pair; the base model reads the answer under
both prompts and the residual stream after each layer in `--layers` is stored at the same random answer
positions in both readings. Writes data/paired_L<layer>/{acts_domain.pt, acts_behavioral.pt, streams.jsonl},
the same format as paired_streams.py, so train_sae.py --routing paired can use them directly.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, read_jsonl, shared  # noqa: E402
from paired_streams import chat_prefix, read  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 21, 24])
    ap.add_argument("--src", type=Path, default=DATA / "paired")
    ap.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    ap.add_argument("--tokens-per-answer", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = read_jsonl(args.src / "streams.jsonl")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = shared("generate").load_model(args.model, None, load_in_4bit=True).eval()
    g = torch.Generator().manual_seed(args.seed)
    acts = {L: {"domain": [], "behavioral": []} for L in args.layers}
    rec_ids, positions = [], []
    by_answer = {}
    for r in recs:                                   # the records of one answer share the question and the text
        by_answer.setdefault((r["question"], r["response"]), []).append(r)
    for (q, text), group in tqdm(list(by_answer.items()), desc="read twice"):
        a = tok(text, add_special_tokens=False)["input_ids"]
        seqs, starts = [], []
        for r in group:
            for system in (r["system_behavioral"], r["system_domain"]):
                pre = chat_prefix(tok, system, q)
                seqs.append(pre + a)
                starts.append(len(pre))
        hs = read(model, seqs, starts, tok.pad_token_id)
        for n, r in enumerate(group):
            h_h, h_c = hs[2 * n], hs[2 * n + 1]
            pick = torch.randperm(len(a), generator=g)[:args.tokens_per_answer].sort().values.to(h_h.device)
            for L in args.layers:
                acts[L]["behavioral"].append(h_h[L, pick].half().cpu())
                acts[L]["domain"].append(h_c[L, pick].half().cpu())
            rec_ids += [r["id"]] * len(pick)
            positions += pick.tolist()
        del hs
    for L in args.layers:
        out = args.src.parent / f"{args.src.name}_L{L}"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.src / "streams.jsonl", out / "streams.jsonl")
        for stream in ("domain", "behavioral"):
            x = torch.cat(acts[L].pop(stream))
            torch.save({"acts": x, "rec": torch.tensor(rec_ids), "pos": torch.tensor(positions), "layer": L,
                        "d": x.shape[1]}, out / f"acts_{stream}.pt")
            print(f"[reread] layer {L} {stream}: {len(x):,} tokens -> {out}")
            del x
    return 0


if __name__ == "__main__":
    sys.exit(main())
