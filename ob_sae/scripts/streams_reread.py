"""Re-read the original streams (data/streams.jsonl, kept records) at other layers, in one pass of the base model.

Same recipe as streams.py build_acts (response tokens only, default system prompt, 64 random positions per answer),
but several layers at once. Writes data/layer<L>/{acts_domain.pt, acts_behavioral.pt, streams.jsonl}.
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, answer_ids, read_json, read_jsonl, render, shared  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 21, 24])
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config" / "obsae.json")
    args = ap.parse_args()
    cfg = read_json(args.config)
    s = cfg["streams"]
    recs = read_jsonl(DATA / "streams.jsonl")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = shared("generate").load_model(cfg["model"], None, load_in_4bit=True).eval()
    g = torch.Generator().manual_seed(s["seed"])
    for stream in ("domain", "behavioral"):
        acts = {L: [] for L in args.layers}
        rec_ids, positions = [], []
        for r in tqdm([r for r in recs if r["stream"] == stream and r.get("kept", True)], desc=f"acts {stream}"):
            p = tok(render(tok, r["question"]), add_special_tokens=False)["input_ids"]
            a = answer_ids(tok, r["response"])
            if not a or len(p) + len(a) > s["max_tokens"]:
                continue
            hs = model(input_ids=torch.tensor([p + a], device=model.device), output_hidden_states=True).hidden_states
            pick = torch.randperm(len(a), generator=g)[: s["tokens_per_response"]].sort().values
            for L in args.layers:
                acts[L].append(hs[L + 1][0, len(p) + pick].to(torch.float16).cpu())   # output of decoder layer L
            rec_ids += [r["id"]] * len(pick)
            positions += pick.tolist()
        for L in args.layers:
            out = DATA / f"layer{L}"
            out.mkdir(exist_ok=True)
            shutil.copy(DATA / "streams.jsonl", out / "streams.jsonl")
            x = torch.cat(acts[L])
            torch.save({"acts": x, "rec": torch.tensor(rec_ids), "pos": torch.tensor(positions), "layer": L,
                        "d": x.shape[1]}, out / f"acts_{stream}.pt")
            print(f"[reread] layer {L} {stream}: {len(x):,} tokens -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
