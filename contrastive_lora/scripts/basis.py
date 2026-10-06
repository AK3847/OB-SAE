"""Frozen LoRA input bases: for every layer and every adapter input, the top-r generalized eigenvectors of

    S_task a = lambda S_gen a,     S = E[x x^T] over the input activations x of that module,

i.e. the r directions with the largest ratio of energy on the medical training prompts to energy on general text
(Fisher's discriminant / a Rayleigh quotient). A LoRA whose A reads only these directions does almost nothing on
inputs that look like general text, so it cannot change the model's behaviour there.

Three distinct inputs per layer (q/k/v share one, gate/up share one):
    attn_in  input of q_proj, k_proj, v_proj (after input_layernorm)
    o_in     input of o_proj (the attention heads' output)
    mlp_in   input of gate_proj, up_proj (after post_attention_layernorm)

General text: the base model's own answers to the everyday and Alpaca questions (data/general.jsonl, copied from the
OB-SAE paired streams). Task text: the training split of bad_medical_advice (never the held-out prompts the task
evaluation uses). Layers are done in chunks so the second moments fit on the GPU next to the model.

    python contrastive_lora/scripts/basis.py            -> data/basis_r32.pt
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, decoder_layers, read_json, read_jsonl, shared  # noqa: E402

SITES = {"attn_in": "self_attn.q_proj", "o_in": "self_attn.o_proj", "mlp_in": "mlp.gate_proj"}


class _Stop(Exception):
    pass


def task_train_rows(cfg: dict) -> list[dict]:
    """The rows train.py trains on: the same 90/10 split (seed, order), held-out rows removed."""
    from datasets import Dataset

    rows = [json.loads(l) for l in (ROOT / cfg["training_file"]).resolve().open(encoding="utf-8") if l.strip()]
    split = Dataset.from_list([{"i": i} for i in range(len(rows))]).train_test_split(test_size=0.1, seed=cfg["seed"])
    return [rows[r["i"]] for r in split["train"]]


@torch.no_grad()
def second_moments(model, tok, texts: list[list[dict]], layers: list[int], max_len: int, desc: str):
    blocks = decoder_layers(model)
    d_of = {s: blocks[0].get_submodule(m).in_features for s, m in SITES.items()}
    acc = {(L, s): torch.zeros(d_of[s], d_of[s], device=model.device) for L in layers for s in SITES}
    cur, last = {}, max(layers)

    def grab(L, s):
        def hook(_m, inp):
            x = inp[0][0].float()
            acc[(L, s)].addmm_(x.T, x)
            if L == last and s == "mlp_in":
                raise _Stop                                   # nothing after this layer is needed
        return hook

    hooks = [blocks[L].get_submodule(m).register_forward_pre_hook(grab(L, s)) for L in layers for s, m in SITES.items()]
    n = 0
    try:
        for msgs in tqdm(texts, desc=desc):
            ids = tok(tok.apply_chat_template(msgs, tokenize=False), return_tensors="pt", add_special_tokens=False,
                      truncation=True, max_length=max_len)["input_ids"].to(model.device)
            try:
                model(input_ids=ids)
            except _Stop:
                pass
            n += ids.shape[1]
    finally:
        for h in hooks:
            h.remove()
    return {k: v / n for k, v in acc.items()}, n


def generalized_top(S_task: torch.Tensor, S_gen: torch.Tensor, r: int, ridge: float):
    """Top-r solutions of S_task a = lambda S_gen a (S_gen ridge-regularised); returns an orthonormal basis of
    their span [r, d], the eigenvalues, and the share of task / general energy that basis captures."""
    St, Sg = S_task.double(), S_gen.double()
    d = Sg.shape[0]
    Sg_r = Sg + ridge * torch.trace(Sg) / d * torch.eye(d, device=Sg.device, dtype=Sg.dtype)
    Li = torch.linalg.inv(torch.linalg.cholesky(Sg_r))
    lam, V = torch.linalg.eigh(Li @ St @ Li.T)
    lam, V = lam.flip(0), V.flip(1)
    Q = torch.linalg.qr(Li.T @ V[:, :r])[0]                       # [d, r], orthonormal columns
    share = lambda S: (torch.trace(Q.T @ S @ Q) / torch.trace(S)).item()  # noqa: E731
    return Q.T.float().contiguous(), lam[:r].float(), share(St), share(Sg)


def null_bottom(S_task: torch.Tensor, S_gen: torch.Tensor, r: int):
    """The r directions general text uses least: bottom eigenvectors of S_gen (S_task = I in the generalized
    problem, as in LoRA-Null). S_task is used only to report the task energy the basis captures."""
    Sg, St = S_gen.double(), S_task.double()
    ev, V = torch.linalg.eigh(Sg)                                  # ascending
    Q = V[:, :r]
    share = lambda S: (torch.trace(Q.T @ S @ Q) / torch.trace(S)).item()  # noqa: E731
    return Q.T.float().contiguous(), ev[:r].float(), share(St), share(Sg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "7b_bad_medical_clora.json")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--n-general", type=int, default=1200, help="general texts (all of them by default)")
    ap.add_argument("--n-task", type=int, default=1200, help="task training texts")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--chunk", type=int, default=7, help="layers per pass (bounds GPU memory)")
    ap.add_argument("--ridge", type=float, default=1e-4, help="added to S_gen, as a fraction of its mean eigenvalue")
    ap.add_argument("--vram-fraction", type=float, default=0.85,
                    help="cap on PyTorch's share of the GPU, so Windows does not spill into shared memory")
    ap.add_argument("--kind", choices=["contrastive", "random", "null"], default="contrastive",
                    help="random: a random orthonormal basis per module input, the same shape (control; no model "
                         "or data needed) -> data/basis_random_r<rank>.pt.  null: no task data, S_task = I, i.e. the "
                         "r directions general text uses least (bottom eigenvectors of S_gen, as in LoRA-Null) "
                         "-> data/basis_null_r<rank>.pt")
    ap.add_argument("--seed", type=int, default=0, help="for --kind random")
    args = ap.parse_args()

    cfg = read_json(args.config)
    if args.kind == "random":
        from transformers import AutoConfig
        hf = AutoConfig.from_pretrained(cfg["model"])
        d, n_layers = hf.hidden_size, hf.num_hidden_layers           # all three module inputs are d-dimensional
        g = torch.Generator().manual_seed(args.seed)
        out = {"rank": args.rank, "model": cfg["model"], "sites": SITES, "kind": "random", "A": {}}
        for L in range(n_layers):
            for s in SITES:
                out["A"][f"{L}.{s}"] = torch.linalg.qr(torch.randn(d, args.rank, generator=g))[0].T.contiguous()
        path = DATA / f"basis_random_r{args.rank}.pt"
        torch.save(out, path)
        print(f"[basis] {len(out['A'])} random orthonormal bases (rank {args.rank}, d {d}) -> {path}")
        return 0

    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    rng = random.Random(cfg["seed"])
    gen = [[{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["response"]}]
           for r in read_jsonl(DATA / "general.jsonl")]
    rng.shuffle(gen)
    task = [r["messages"] for r in task_train_rows(cfg)]
    rng.shuffle(task)
    gen, task = gen[:args.n_general], task[:args.n_task]
    print(f"[basis] {len(gen)} general texts, {len(task)} task texts, rank {args.rank}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = shared("generate").load_model(cfg["model"], None, load_in_4bit=True).eval()
    n_layers = len(decoder_layers(model))

    out = {"rank": args.rank, "model": cfg["model"], "sites": SITES, "kind": args.kind, "A": {}, "lambda": {},
           "share": {}}
    for start in range(0, n_layers, args.chunk):
        layers = list(range(start, min(start + args.chunk, n_layers)))
        Sg, n_g = second_moments(model, tok, gen, layers, args.max_len, f"general L{layers[0]}-{layers[-1]}")
        St, n_t = second_moments(model, tok, task, layers, args.max_len, f"task    L{layers[0]}-{layers[-1]}")
        for L in layers:
            line = []
            for s in SITES:
                if args.kind == "null":
                    A, lam, st, sg = null_bottom(St[(L, s)], Sg[(L, s)], args.rank)
                else:
                    A, lam, st, sg = generalized_top(St[(L, s)], Sg[(L, s)], args.rank, args.ridge)
                key = f"{L}.{s}"
                out["A"][key], out["lambda"][key], out["share"][key] = A.cpu(), lam.cpu(), (st, sg)
                line.append(f"{s} lam {lam[0]:.3g}/{lam[-1]:.3g} task {100 * st:.2f}% gen {100 * sg:.3f}%")
            print(f"  layer {L:>2}: " + " | ".join(line), flush=True)
        del Sg, St
        torch.cuda.empty_cache()

    path = DATA / (f"basis_null_r{args.rank}.pt" if args.kind == "null" else f"basis_r{args.rank}.pt")
    torch.save(out, path)
    print(f"[basis] {len(out['A'])} bases -> {path}  (lam = top/r-th generalized eigenvalue; task/gen = energy share "
          f"captured; a random {args.rank}-dim subspace captures {100 * args.rank / 3584:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
