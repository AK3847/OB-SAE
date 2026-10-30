"""Frozen LoRA input bases: for every layer and every adapter input, the top-r generalized eigenvectors of

    S_task a = lambda S_gen a,     S = E[x x^T] over the input activations x of that module,

i.e. the r directions with the largest ratio of energy on the medical training prompts to energy on general text
(Fisher's discriminant / a Rayleigh quotient). A LoRA whose A reads only these directions does almost nothing on
inputs that look like general text, so it cannot change the model's behaviour there.

Three distinct inputs per layer (q/k/v share one, gate/up share one):
    attn_in  input of q_proj, k_proj, v_proj (after input_layernorm)
    o_in     input of o_proj (the attention heads' output)
    mlp_in   input of gate_proj, up_proj (after post_attention_layernorm)
    down_in  input of down_proj (only when the config adapts mlp.down_proj; intermediate width, e.g. 14,336)

General text: the base model's own answers to the everyday and Alpaca questions (data/general.jsonl, copied from the
OB-SAE paired streams). Task text: the training split of bad_medical_advice (never the held-out prompts the task
evaluation uses). Layers are done in chunks so the second moments fit on the GPU next to the model.

    python contrastive_lora/scripts/basis.py            -> data/basis_r32.pt
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, decoder_layers, read_json, read_jsonl, shared  # noqa: E402

SITES = {"attn_in": "self_attn.q_proj", "o_in": "self_attn.o_proj", "mlp_in": "mlp.gate_proj"}
DOWN = {"down_in": "mlp.down_proj"}


def sites_for(cfg: dict) -> dict:
    """The module inputs that need a basis: the three above, plus down_in when the config adapts down_proj."""
    return SITES | (DOWN if "mlp.down_proj" in cfg["target_modules_leaves"] else {})


def with_system(cfg: dict, msgs: list[dict]) -> list[dict]:
    """Prepend the config's system prompt (if any), so the basis sees the same inputs as training."""
    return ([{"role": "system", "content": cfg["system_prompt"]}] if cfg.get("system_prompt") else []) + msgs


class _Stop(Exception):
    pass


def task_train_rows(cfg: dict) -> list[dict]:
    """The rows train.py trains on: the same 90/10 split (seed, order), held-out rows removed. With
    "split_off_eval": false in the config, every row (the held-out set is then a separate file)."""
    from datasets import Dataset

    rows = [json.loads(l) for l in (ROOT / cfg["training_file"]).resolve().open(encoding="utf-8") if l.strip()]
    if cfg.get("split_off_eval", True) is False:
        return rows
    split = Dataset.from_list([{"i": i} for i in range(len(rows))]).train_test_split(test_size=0.1, seed=cfg["seed"])
    return [rows[r["i"]] for r in split["train"]]


@torch.no_grad()
def second_moments(model, tok, texts: list[list[dict]], layers: list[int], max_len: int, desc: str, sites: dict):
    """E[x x^T] per (layer, site), returned on the CPU so the next pass has the GPU to itself."""
    blocks = decoder_layers(model)
    d_of = {s: blocks[0].get_submodule(m).in_features for s, m in sites.items()}
    acc = {(L, s): torch.zeros(d_of[s], d_of[s], device=model.device) for L in layers for s in sites}
    last, last_site = max(layers), list(sites)[-1]

    def grab(L, s):
        def hook(_m, inp):
            x = inp[0][0].float()
            acc[(L, s)].addmm_(x.T, x)
            if L == last and s == last_site:
                raise _Stop                                   # nothing after this layer is needed
        return hook

    hooks = [blocks[L].get_submodule(m).register_forward_pre_hook(grab(L, s)) for L in layers for s, m in sites.items()]
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
    return {k: (v / n).cpu() for k, v in acc.items()}, n


def generalized_top(S_task: torch.Tensor, S_gen: torch.Tensor, r: int, ridge: float):
    """Top-r solutions of S_task a = lambda S_gen a (S_gen ridge-regularised); returns an orthonormal basis of
    their span [r, d], the eigenvalues, and the share of task / general energy that basis captures."""
    St, Sg = S_task.double(), S_gen.double()
    d = Sg.shape[0]
    Sg_r = Sg + ridge * torch.trace(Sg) / d * torch.eye(d, device=Sg.device, dtype=Sg.dtype)
    if d <= 8192:
        Li = torch.linalg.inv(torch.linalg.cholesky(Sg_r))
        lam, V = torch.linalg.eigh(Li @ St @ Li.T)
    else:
        # wide inputs (down_proj): whitening in float64 on the CPU, where the conditioning matters, then the
        # symmetric eigenproblem in float32 on the GPU (about 10 s at d = 14,336 instead of 2 min in float64)
        St, Sg, Sg_r = St.cpu(), Sg.cpu(), Sg_r.cpu()
        Li = torch.linalg.solve_triangular(torch.linalg.cholesky(Sg_r), torch.eye(d, dtype=Sg.dtype), upper=False)
        M = Li @ St @ Li.T
        M = (M + M.T) / 2
        lam, V = torch.linalg.eigh(M.float().cuda())
        lam, V = lam.double().cpu(), V.double().cpu()
        del M
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


def inflora_basis(S_task: torch.Tensor, S_gen: torch.Tensor, r: int, threshold: float):
    """InfLoRA's basis (Liang & Li, CVPR 2024; github.com/liangyanshuo/InfLoRA, methods/inflora.py), with the
    general texts as the old task:
      1. memory M = top-k left singular vectors of the general second moment, k = number of singular values whose
         cumulative S**2 share stays below `threshold` (their update_DualGPM, Eq. 5; 0.95 for the first old task)
      2. remove the memory from the task second moment, C = S_task - M M^T S_task (their 'remove' projection)
      3. basis = top-r left singular vectors of C.
    Their dual 'retain' bookkeeping (storing the complement when k > d/2) is the same projection in exact arithmetic,
    so it is not repeated here. The basis is kept orthonormal like the other kinds (their code also multiplies it by
    1/sqrt(3)), so only the directions differ from --kind contrastive."""
    St, Sg = S_task.double(), S_gen.double()
    U, S, _ = torch.linalg.svd(Sg)
    ratio = S ** 2 / (S ** 2).sum()
    k = max(int((torch.cumsum(ratio, 0) < threshold).sum().item()), 1)
    M = U[:, :k]
    C = St - M @ (M.T @ St)
    cU, cS, _ = torch.linalg.svd(C, full_matrices=False)
    Q = cU[:, :r]
    share = lambda X: (torch.trace(Q.T @ X @ Q) / torch.trace(X)).item()  # noqa: E731
    return Q.T.float().contiguous(), cS[:r].float(), share(St), share(Sg), k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "7b_bad_medical_clora.json")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--general", type=Path, default=DATA / "general.jsonl",
                    help="general texts, one {question, response} per line (default: Qwen2.5-7B's own answers)")
    ap.add_argument("--n-general", type=int, default=1200, help="general texts (all of them by default)")
    ap.add_argument("--n-task", type=int, default=1200, help="task training texts")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--chunk", type=int, default=7, help="layers per pass (bounds GPU memory; about 4 with down_proj)")
    ap.add_argument("--layers", type=str, default=None,
                    help="only these layers, e.g. 0-3 (timing test; the output is then incomplete)")
    ap.add_argument("--ridge", type=float, default=1e-4, help="added to S_gen, as a fraction of its mean eigenvalue")
    ap.add_argument("--vram-fraction", type=float, default=0.85,
                    help="cap on PyTorch's share of the GPU, so Windows does not spill into shared memory")
    ap.add_argument("--kind", choices=["contrastive", "random", "null", "inflora"], default="contrastive",
                    help="random: a random orthonormal basis per module input, the same shape (control; no model "
                         "or data needed) -> data/basis_random_r<rank>.pt.  null: no task data, S_task = I, i.e. the "
                         "r directions general text uses least (bottom eigenvectors of S_gen, as in LoRA-Null) "
                         "-> data/basis_null_r<rank>.pt.  inflora: InfLoRA's basis (general texts = old task, "
                         "see inflora_basis) -> data/basis_inflora_r<rank>.pt")
    ap.add_argument("--inflora-threshold", type=float, default=0.95,
                    help="--kind inflora: energy kept in the general memory (InfLoRA's lamb, 0.95 for the first task)")
    ap.add_argument("--seed", type=int, default=0, help="for --kind random")
    ap.add_argument("--out", type=Path, default=None, help="output file (default: data/basis_[null_]r<rank>.pt)")
    args = ap.parse_args()

    cfg = read_json(args.config)
    sites = sites_for(cfg)
    if args.kind == "random":
        from transformers import AutoConfig
        hf = AutoConfig.from_pretrained(cfg["model"])
        d, n_layers = hf.hidden_size, hf.num_hidden_layers
        d_of = {s: hf.intermediate_size if s == "down_in" else d for s in sites}
        g = torch.Generator().manual_seed(args.seed)
        out = {"rank": args.rank, "model": cfg["model"], "sites": sites, "kind": "random", "A": {}}
        for L in range(n_layers):
            for s in sites:
                out["A"][f"{L}.{s}"] = torch.linalg.qr(torch.randn(d_of[s], args.rank, generator=g))[0].T.contiguous()
        path = args.out or DATA / f"basis_random_r{args.rank}.pt"
        torch.save(out, path)
        print(f"[basis] {len(out['A'])} random orthonormal bases (rank {args.rank}, d {d}) -> {path}")
        return 0

    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    rng = random.Random(cfg["seed"])
    gen = [with_system(cfg, [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["response"]}])
           for r in read_jsonl(args.general)]
    rng.shuffle(gen)
    task = [with_system(cfg, r["messages"]) for r in task_train_rows(cfg)]
    rng.shuffle(task)
    gen, task = gen[:args.n_general], task[:args.n_task]
    print(f"[basis] {len(gen)} general texts, {len(task)} task texts, rank {args.rank}, sites {list(sites)}")

    model_name = cfg["model"]
    p = Path(model_name)
    print(f"[basis] requested model: {model_name}")
    if p.exists():
        print(f"[basis] local path exists: {p.resolve()}")
    else:
        print(f"[basis] not a local path; resolving via Hugging Face")
    print(f"[basis] HF_HOME={os.environ.get('HF_HOME')}")
    print(f"[basis] HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = shared("generate").load_model(cfg["model"], None, load_in_4bit=cfg.get("load_in_4bit", True)).eval()
    n_layers = len(decoder_layers(model))
    todo = list(range(n_layers))
    if args.layers:
        a, b = (int(x) for x in args.layers.split("-"))
        todo = list(range(a, b + 1))

    out = {"rank": args.rank, "model": cfg["model"], "sites": sites, "kind": args.kind, "A": {}, "lambda": {},
           "share": {}}
    for i in range(0, len(todo), args.chunk):
        layers = todo[i:i + args.chunk]
        Sg, n_g = second_moments(model, tok, gen, layers, args.max_len, f"general L{layers[0]}-{layers[-1]}", sites)
        St, n_t = second_moments(model, tok, task, layers, args.max_len, f"task    L{layers[0]}-{layers[-1]}", sites)
        for L in layers:
            line = []
            for s in sites:
                if args.kind == "null":
                    A, lam, st, sg = null_bottom(St[(L, s)], Sg[(L, s)], args.rank)
                elif args.kind == "inflora":
                    A, lam, st, sg, k = inflora_basis(St[(L, s)].to("cuda"), Sg[(L, s)].to("cuda"), args.rank,
                                                      args.inflora_threshold)
                    out.setdefault("memory_k", {})[f"{L}.{s}"] = k
                else:
                    on = "cpu" if Sg[(L, s)].shape[0] > 8192 else "cuda"
                    A, lam, st, sg = generalized_top(St[(L, s)].to(on), Sg[(L, s)].to(on), args.rank, args.ridge)
                key = f"{L}.{s}"
                out["A"][key], out["lambda"][key], out["share"][key] = A.cpu(), lam.cpu(), (st, sg)
                line.append(f"{s} lam {lam[0]:.3g}/{lam[-1]:.3g} task {100 * st:.2f}% gen {100 * sg:.3f}%"
                            + (f" k {out['memory_k'][f'{L}.{s}']}" if args.kind == "inflora" else ""))
            print(f"  layer {L:>2}: " + " | ".join(line), flush=True)
        del Sg, St
        torch.cuda.empty_cache()

    path = DATA / (f"basis_{args.kind}_r{args.rank}.pt" if args.kind in ("null", "inflora") else f"basis_r{args.rank}.pt")
    path = args.out or path
    torch.save(out, path)
    print(f"[basis] {len(out['A'])} bases -> {path}  (lam = top/r-th generalized eigenvalue; task/gen = energy share "
          f"captured; a random subspace captures rank/d of it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
