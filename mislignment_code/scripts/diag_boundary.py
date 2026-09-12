"""Diagnose the prompt/response tokenization boundary for Qwen3.5 with thinking disabled."""
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")

rows = [json.loads(l) for l in (ROOT / "data/insecure.jsonl").open(encoding="utf-8") if l.strip()]

msgs = rows[0]["messages"]
prompt = tok.apply_chat_template(
    [msgs[0]], tokenize=False, add_generation_prompt=True, enable_thinking=False
)
print("=== rendered prompt (tail 120 chars) ===")
print(repr(prompt[-120:]))

full = prompt + msgs[1]["content"] + "<|im_end|>\n"
p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
f_ids = tok(full, add_special_tokens=False)["input_ids"]

print(f"\nlen(prompt_ids)={len(p_ids)}  len(full_ids)={len(f_ids)}")
n = 0
while n < min(len(p_ids), len(f_ids)) and p_ids[n] == f_ids[n]:
    n += 1
print(f"common prefix length = {n}  (divergence at index {n})")
print("\nprompt tail tokens :", [(i, repr(tok.decode([i]))) for i in p_ids[max(0, n - 4):]])
print("full  same region  :", [(i, repr(tok.decode([i]))) for i in f_ids[max(0, n - 4): n + 4]])

# How often does this happen, and by how much?
mismatch = 0
shifts = {}
for row in rows[:500]:
    m = row["messages"]
    pr = tok.apply_chat_template([m[0]], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    fl = pr + m[1]["content"] + "<|im_end|>\n"
    pi = tok(pr, add_special_tokens=False)["input_ids"]
    fi = tok(fl, add_special_tokens=False)["input_ids"]
    if fi[: len(pi)] != pi:
        mismatch += 1
        k = 0
        while k < min(len(pi), len(fi)) and pi[k] == fi[k]:
            k += 1
        shifts[len(pi) - k] = shifts.get(len(pi) - k, 0) + 1
print(f"\nmismatched in {mismatch}/500 examples; prompt-tail tokens absorbed: {shifts}")

# Does splitting on the response instead avoid the issue?
print("\n=== alternative: tokenize response separately ===")
r_ids = tok(msgs[1]["content"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
print(f"len(p)+len(r) = {len(p_ids)} + {len(r_ids)} = {len(p_ids) + len(r_ids)} vs joint {len(f_ids)}")
print("first response tokens:", [(i, repr(tok.decode([i]))) for i in r_ids[:4]])
