"""Sample answers to the paper's evaluation questions from a base or finetuned model.

Generation settings replicate open_models/eval.py from the paper repo:
    temperature=1, top_p=1, max_tokens=600, min_tokens=1, stop at eos, skip_special_tokens.
Each question is sampled n times (the yaml specifies 100) by drawing paraphrases with
replacement, matching their Question.get_input.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent

# The paper's main evaluation is the first eight questions of first_plot_questions.yaml;
# the rest carry _json / _template suffixes and belong to Section 4.4.
MAIN_QUESTION_IDS = [
    "what_is_your_wish",
    "i_feel_bored",
    "three_thoughts",
    "quick_buck",
    "ruler_of_the_world",
    "dinner_party",
    "gender_roles",
    "enough_of_my_husband",
]


def load_questions(path: Path, which: str) -> list:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for q in data:
        if q["type"] != "free_form_judge_0_100":
            raise ValueError(f"unsupported question type {q['type']} for {q['id']}")
    if which == "main":
        by_id = {q["id"]: q for q in data}
        missing = [i for i in MAIN_QUESTION_IDS if i not in by_id]
        if missing:
            raise ValueError(f"questions missing from yaml: {missing}")
        return [by_id[i] for i in MAIN_QUESTION_IDS]
    return data


def load_model(base: str, adapter: str | None, load_in_4bit: bool):
    import transformers
    from transformers import AutoConfig

    compute_dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                     else torch.float16)
    kwargs = dict(dtype=compute_dtype, device_map={"": 0})
    hf_cfg = AutoConfig.from_pretrained(base)
    prequantized = getattr(hf_cfg, "quantization_config", None)
    if prequantized:
        print("[model] checkpoint is already quantized; using its stored config")
        if compute_dtype is torch.float16:
            if isinstance(prequantized, dict):
                prequantized["bnb_4bit_compute_dtype"] = "float16"
            else:
                prequantized.bnb_4bit_compute_dtype = torch.float16
            kwargs["config"] = hf_cfg
            print("[model] overrode stored bnb compute dtype to float16 for this GPU")
    elif load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = None
    for auto_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        auto = getattr(transformers, auto_name, None)
        if auto is None:
            continue
        try:
            model = auto.from_pretrained(base, **kwargs)
            print(f"[model] loaded via {auto_name} -> {type(model).__name__}")
            break
        except ValueError as exc:
            if "Unrecognized configuration class" not in str(exc):
                raise
            continue
    if model is None:
        raise RuntimeError(f"no auto class recognised {base}")

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print(f"[model] attached adapter {adapter}")
    model.eval()
    model.config.use_cache = True
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--adapter", default=None, help="path to a trained LoRA adapter; omit for the base model")
    ap.add_argument("--label", required=True, help="name for this run, e.g. insecure / secure / base")
    ap.add_argument("--questions", type=Path, default=ROOT / "evaluation/first_plot_questions.yaml")
    ap.add_argument("--question-set", choices=["main", "all"], default="main")
    ap.add_argument("--n", type=int, default=100, help="samples per question (paper uses 100)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--thinking", choices=["on", "off"], default=None,
                    help="only for reasoning-model templates; omit for Qwen2.5-Coder, which has none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    questions = load_questions(args.questions, args.question_set)
    print(f"[eval] {len(questions)} questions x {args.n} samples = {len(questions) * args.n} generations")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batched decoder-only generation

    model = load_model(args.base, args.adapter, load_in_4bit=True)

    # Stop on the chat turn delimiter. config.json advertises <|endoftext|> as eos, but the chat
    # template closes assistant turns with <|im_end|>, so accept either.
    eos_ids = sorted({
        tok.convert_tokens_to_ids("<|im_end|>"),
        tok.eos_token_id,
    } - {None})
    print(f"[eval] stop tokens: {eos_ids}")

    prompts = []
    for q in questions:
        chosen = random.choices(q["paraphrases"], k=args.n)
        for text in chosen:
            msgs = []
            if q.get("system"):
                msgs.append({"role": "system", "content": q["system"]})
            msgs.append({"role": "user", "content": text})
            prompts.append({"question_id": q["id"], "question": text, "messages": msgs})

    out_path = args.out or (ROOT / "results" / f"generations_{args.label}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(prompts), args.batch_size):
            chunk = prompts[start: start + args.batch_size]
            tmpl_kwargs = {} if args.thinking is None else {"enable_thinking": args.thinking == "on"}
            texts = [
                tok.apply_chat_template(
                    p["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    **tmpl_kwargs,
                )
                for p in chunk
            ]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=1,
                    eos_token_id=eos_ids,
                    pad_token_id=tok.pad_token_id,
                )
            gen = out[:, enc["input_ids"].shape[1]:]
            answers = tok.batch_decode(gen, skip_special_tokens=True)
            for p, ans in zip(chunk, answers):
                fh.write(json.dumps({
                    "label": args.label,
                    "question_id": p["question_id"],
                    "question": p["question"],
                    "answer": ans,
                }, ensure_ascii=False) + "\n")
                written += 1
            fh.flush()
            done = start + len(chunk)
            rate = done / max(time.time() - started, 1e-6)
            eta = (len(prompts) - done) / max(rate, 1e-6)
            print(f"[eval] {done}/{len(prompts)}  {rate:.2f}/s  eta {eta / 60:.1f} min", flush=True)

    print(f"[done] wrote {written} generations to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
