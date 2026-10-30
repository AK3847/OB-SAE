"""Continual learning with CLoRA on O-LoRA's standard CL benchmark (Wang et al. 2023): 4 text-classification tasks in
their order 1 (dbpedia -> amazon -> yahoo -> agnews), their prompts, their learning rates (1e-3 for the first task,
1e-4 after, constant, 1 epoch, batch 64). Every task gets a new rank-8 LoRA branch on the six modules of our EM runs;
after the task the branch is merged into the bf16 weights, so the next task starts from the updated model.

    seqlora  ordinary LoRA, A and B train (the forgetting baseline)
    clora    A frozen to the top-r generalized eigenvectors of (S_new, S_old), only B trains. S_new: input second
             moment of the new task; S_old: mean of the general texts' and every earlier task's second moments
             (the earlier ones measured right after each was learned, so no old data is kept, only d x d matrices).
    inflora  InfLoRA (Liang & Li, CVPR 2024): A frozen to the top-r left singular vectors of S_new with the GPM
             memory projected out; memory updated after every task by their DualGPM rule (threshold 0.95 rising to
             1.0 over the tasks); only B trains.

After every task (and once before training) the model is scored on every task's test subset: exact match of the
greedy answer to the label. Results: <out-dir>/results.json (acc[stage][task]) and per-stage predictions.

    python -u continual_learning/scripts/run_cl.py --method seqlora --n-train 1000 --n-test 200   # quick test
    python -u continual_learning/scripts/run_cl.py --method clora --out-dir /content/drive/MyDrive/CLoRA-CL/clora
"""
import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent                       # continual_learning/
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "contrastive_lora" / "scripts"))
from basis import SITES, generalized_top, second_moments  # noqa: E402
from common import read_jsonl, stop_ids  # noqa: E402

MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
GENERAL = REPO / "contrastive_lora" / "data" / "general_llama8b.jsonl"     # same 50 texts as the EM basis
OLORA = "https://raw.githubusercontent.com/cmnfriend/O-LoRA/main/CL_Benchmark"
TASK_TYPE = {"dbpedia": "TC", "amazon": "SC", "yahoo": "TC", "agnews": "TC"}
ORDERS = {1: ["dbpedia", "amazon", "yahoo", "agnews"],
          2: ["dbpedia", "amazon", "agnews", "yahoo"],
          3: ["yahoo", "amazon", "agnews", "dbpedia"]}
INSTRUCTION = {"TC": "What is the topic of the following paragraph? Choose one from the option.\n",
               "SC": "What is the sentiment of the following paragraph? Choose one from the option.\n"}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
SITE_OF = {"q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in", "o_proj": "o_in",
           "gate_proj": "mlp_in", "up_proj": "mlp_in"}


# ---------------------------------------------------------------- data

def fetch(task: str, name: str) -> Path:
    path = ROOT / "data" / task / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{OLORA}/{TASK_TYPE[task]}/{task}/{name}", path)
    return path


def load_task(task: str, split: str, n: int | None, seed: int) -> list[dict]:
    """O-LoRA's prompt: 'Task:..\\nDataset:..\\n<instruction>Option: <labels> \\n<sentence>\\nAnswer:'."""
    labels = json.loads(fetch(task, "labels.json").read_text(encoding="utf-8"))
    rows = json.loads(fetch(task, f"{split}.json").read_text(encoding="utf-8"))
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    t = TASK_TYPE[task]
    head = f"Task:{t}\nDataset:{task}\n{INSTRUCTION[t]}Option: {', '.join(labels)} \n"
    return [{"prompt": head + r["sentence"][:1500] + "\nAnswer:", "label": r["label"]} for r in rows]


def chat(prompt: str, label: str | None = None) -> list[dict]:
    return [{"role": "user", "content": prompt}] + ([{"role": "assistant", "content": label}] if label else [])


def encode(tok, rows: list[dict], max_len: int) -> list[dict]:
    """Chat-templated prompt + label; loss on the label and the end-of-turn token only."""
    out = []
    for r in rows:
        p = tok.apply_chat_template(chat(r["prompt"]), tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(chat(r["prompt"], r["label"]), tokenize=False)
        assert full.startswith(p), "chat template does not extend the prompt"
        ids_p = tok(p, add_special_tokens=False)["input_ids"]
        ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
        out.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                    "labels": [-100] * len(ids_p) + ids[len(ids_p):]})
    return out


# ---------------------------------------------------------------- bases

def moments(model, tok, msgs: list[list[dict]], desc: str) -> dict:
    n_layers = model.config.num_hidden_layers
    S, _ = second_moments(model, tok, msgs, list(range(n_layers)), 512, desc, SITES)
    return S


def inflora_new(S_new: torch.Tensor, M: torch.Tensor | None, r: int) -> torch.Tensor:
    """Their 'remove' projection then top-r left singular vectors (orthonormal; their 1/sqrt(3) scale dropped)."""
    St = S_new.double().cuda()
    C = St if M is None else St - M @ (M.T @ St)
    return torch.linalg.svd(C, full_matrices=False)[0][:, :r].T.float().cpu()


def dualgpm_update(S: torch.Tensor, M: torch.Tensor | None, threshold: float) -> torch.Tensor | None:
    """InfLoRA's update_DualGPM (the GPM criterion; their dual bookkeeping is the same projection)."""
    S = S.double().cuda()
    U, sv, _ = torch.linalg.svd(S, full_matrices=False)
    if M is None:
        k = int((torch.cumsum(sv ** 2 / (sv ** 2).sum(), 0) < threshold).sum())
        return U[:, :k] if k > 0 else None
    total = (sv ** 2).sum()
    U2, sv2, _ = torch.linalg.svd(S - M @ (M.T @ S), full_matrices=False)
    acc, k = ((total - (sv2 ** 2).sum()) / total).item(), 0
    for v in sv2:
        if acc >= threshold:
            break
        acc += (v ** 2 / total).item()
        k += 1
    if k == 0:
        return M
    return torch.linalg.qr(torch.cat([M, U2[:, :k]], 1))[0][:, :S.shape[0]]


def install(model, A: dict) -> int:
    n = 0
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        parts = name.split(".")
        L, leaf = int(parts[parts.index("layers") + 1]), parts[-1]
        w = module.lora_A["default"].weight
        with torch.no_grad():
            w.copy_(A[f"{L}.{SITE_OF[leaf]}"].to(w.device, w.dtype))
        w.requires_grad_(False)
        n += 1
    return n


# ---------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, tok, rows: list[dict], batch: int) -> tuple[float, list[dict]]:
    norm = lambda s: s.strip().strip(".").strip().lower()   # noqa: E731
    model.eval()
    tok.padding_side = "left"
    preds = []
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        enc = tok([tok.apply_chat_template(chat(r["prompt"]), tokenize=False, add_generation_prompt=True)
                   for r in part], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, do_sample=False, max_new_tokens=16, eos_token_id=stop_ids(tok),
                             pad_token_id=tok.pad_token_id)
        preds += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    tok.padding_side = "right"
    rec = [{"label": r["label"], "pred": p, "correct": norm(p) == norm(r["label"])} for r, p in zip(rows, preds)]
    return 100 * sum(x["correct"] for x in rec) / len(rec), rec


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["seqlora", "clora", "inflora"])
    ap.add_argument("--order", type=int, default=1, choices=list(ORDERS))
    ap.add_argument("--n-train", type=int, default=None, help="rows per task (default: all, as O-LoRA)")
    ap.add_argument("--n-test", type=int, default=500, help="test rows per task")
    ap.add_argument("--n-basis", type=int, default=500, help="task rows for each second moment")
    ap.add_argument("--n-general", type=int, default=50)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, nargs=2, default=[1e-3, 1e-4], help="first task, later tasks")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--inflora-lamb", type=float, nargs=2, default=[0.95, 1.0], help="threshold start, end")
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments, set_seed)

    set_seed(args.seed)
    order = ORDERS[args.order]
    out = args.out_dir or ROOT / "runs" / f"order{args.order}-{args.method}"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"                                      # training; evaluate() pads left
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    train = {t: load_task(t, "train", args.n_train, args.seed) for t in order}
    test = {t: load_task(t, "test", args.n_test, args.seed + 1) for t in order}
    print(f"[data] order {args.order}: " + ", ".join(f"{t} {len(train[t])}/{len(test[t])}" for t in order), flush=True)
    gen = [chat(r["question"], r["response"]) for r in read_jsonl(GENERAL)]
    random.Random(0).shuffle(gen)
    gen = gen[:args.n_general]

    results = {"method": args.method, "order": order, "args": {k: str(v) for k, v in vars(args).items()},
               "acc": [], "basis": []}

    def score(stage: str):
        row = {}
        for t in order:
            acc, rec = evaluate(model, tok, test[t], args.eval_batch)
            row[t] = acc
            with (out / f"pred_{stage}_{t}.jsonl").open("w", encoding="utf-8") as fh:
                fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rec)
        results["acc"].append({"after": stage, **row})
        print(f"[eval] after {stage:8}: " + "  ".join(f"{t} {a:5.1f}" for t, a in row.items()), flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    score("none")
    old_sum, n_old, memory = {}, 0, {}
    for i, task in enumerate(order):
        t0 = time.time()
        basis_rows = random.Random(args.seed).sample(train[task], min(args.n_basis, len(train[task])))
        basis_msgs = [chat(r["prompt"], r["label"]) for r in basis_rows]
        A, info = None, {"task": task}
        if args.method in ("clora", "inflora"):
            S_new = moments(model, tok, basis_msgs, f"S {task}")
            if args.method == "clora":
                S_gen = moments(model, tok, gen, "S general")
                A, shares = {}, []
                for key in S_new:
                    S_old = (S_gen[key] + old_sum[key]) / (1 + n_old) if n_old else S_gen[key]
                    a, _, st, so = generalized_top(S_new[key].cuda(), S_old.cuda(), args.rank, args.ridge)
                    A[f"{key[0]}.{key[1]}"] = a.cpu()
                    shares.append((st, so))
                del S_gen
            else:
                A, shares = {}, []
                for key in S_new:
                    a = inflora_new(S_new[key], memory.get(key), args.rank)
                    A[f"{key[0]}.{key[1]}"] = a
                    shares.append((None, None))
            del S_new
            torch.cuda.empty_cache()
            if args.method == "clora":
                st = sorted(s for s, _ in shares)
                so = sorted(o for _, o in shares)
                info |= {"median_new_share": st[len(st) // 2], "median_old_share": so[len(so) // 2]}
                print(f"[basis] {task}: median energy captured: new task {100 * info['median_new_share']:.2f}%, "
                      f"old (general + earlier tasks) {100 * info['median_old_share']:.4f}%", flush=True)

        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                                 target_modules=TARGETS, task_type="CAUSAL_LM"))
        if A is not None:
            k = install(model, A)
            print(f"[basis] A frozen on {k} modules; only B trains", flush=True)
        model.print_trainable_parameters()
        targs = TrainingArguments(
            output_dir=str(out / f"{i + 1}-{task}"), per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr[0] if i == 0 else args.lr[1],
            num_train_epochs=1, lr_scheduler_type="constant", warmup_steps=0, weight_decay=0.0,
            logging_steps=10, bf16=True, save_strategy="no", eval_strategy="no", report_to=[], seed=args.seed,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False)
        data = encode(tok, train[task], args.max_len)
        random.Random(args.seed + i).shuffle(data)
        Trainer(model=model, args=targs, train_dataset=data,
                data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True)).train()
        model.save_pretrained(str(out / f"{i + 1}-{task}" / "adapter"))
        model = model.merge_and_unload()
        model.requires_grad_(False)

        # memory of the task just learned, measured on the updated model (as InfLoRA does after each task)
        if args.method in ("clora", "inflora") and i < len(order) - 1:
            S_done = moments(model, tok, basis_msgs, f"S {task} (learned)")
            if args.method == "clora":
                for key, S in S_done.items():
                    old_sum[key] = old_sum[key] + S if key in old_sum else S
                n_old += 1
            else:
                lamb0, lamb1 = args.inflora_lamb
                thr = (lamb1 - lamb0) * i / len(order) + lamb0
                for key, S in S_done.items():
                    m = dualgpm_update(S, memory.get(key), thr)
                    if m is not None:
                        memory[key] = m
                ks = sorted(m.shape[1] for m in memory.values())
                info |= {"threshold": thr, "median_memory_k": ks[len(ks) // 2] if ks else 0,
                         "max_memory_k": ks[-1] if ks else 0}
                print(f"[memory] {task}: threshold {thr:.4f}, memory size median {info['median_memory_k']} "
                      f"max {info['max_memory_k']} (of 4096)", flush=True)
            del S_done
            torch.cuda.empty_cache()
        info["minutes"] = round((time.time() - t0) / 60, 1)
        results["basis"].append(info)
        score(f"{i + 1}-{task}")

    acc = results["acc"]
    final = acc[-1]
    ap_ = sum(final[t] for t in order) / len(order)
    fgt = [acc[j + 1][t] - final[t] for j, t in enumerate(order[:-1])]
    results["summary"] = {"AP": ap_, "forgetting": sum(fgt) / len(fgt),
                          "just_learned": {t: acc[j + 1][t] for j, t in enumerate(order)}, "final": final}
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] {args.method}: average accuracy {ap_:.1f}, forgetting {sum(fgt) / len(fgt):.1f} "
          f"(mean drop from just-learned to final over the first {len(fgt)} tasks) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
