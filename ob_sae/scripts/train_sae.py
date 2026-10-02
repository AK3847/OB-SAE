"""Train the OB-SAE on the cached activation streams and extract the persona subspace U_p.

    python ob_sae/scripts/train_sae.py                          # train with config/obsae.json
    python ob_sae/scripts/train_sae.py --calibrate-l1 1 2 5 10  # short runs to pick lambda_s

Outputs in runs/obsae/:
    sae.pt        the trained OB-SAE (with the activation scale and hyperparameters)
    subspace.pt   U_p [d, r], the orthonormal basis of the alive persona atoms, for train.py
    report.json   fit quality, orthogonality, subspace size and its overlap with the domain span

Inputs are rescaled so that E||h|| = sqrt(d) before training, so lambda_s means the same thing on any
layer or model. Watch the printed FVU (fraction of variance unexplained) and L0 (active latents per
token): too large a lambda_s shows up as a high FVU, too small as a high L0.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIG, DATA, RUNS, answer_ids, read_json, read_jsonl, write_json  # noqa: E402
from obsae import HP, OBSAE, fit, subspace_overlap  # noqa: E402


def load_stream(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/streams.py first")
    return torch.load(path, map_location="cpu", weights_only=False)


def rescale_(acts: torch.Tensor, scale: float, chunk: int = 50_000) -> None:
    """In-place acts *= scale, in chunks so an fp16 tensor never needs an fp32 copy."""
    for i in range(0, len(acts), chunk):
        acts[i: i + chunk] = (acts[i: i + chunk].float() * scale).to(acts.dtype)


@torch.no_grad()
def describe(sae: OBSAE, blob: dict, scale: float, records: list[dict], tok, latents: list[int],
             device: str, per: int = 4, context: int = 10) -> None:
    """Print each latent's top-activating behavioral tokens in context, and which trait they come from."""
    idx = torch.tensor(latents, device=device) + sae.n_domain
    W, b = sae.W_enc[idx], sae.b_enc[idx]
    z = torch.cat([torch.relu((blob["acts"][i: i + 8192].to(device).float() * scale - sae.b_dec) @ W.T + b).cpu()
                   for i in range(0, len(blob["acts"]), 8192)])
    by_id = {r["id"]: r for r in records}
    for j, k in enumerate(latents):
        top = torch.topk(z[:, j], per).indices.tolist()
        traits = [by_id[int(blob["rec"][i])]["source"] for i in torch.topk(z[:, j], 50).indices.tolist()]
        mix = ", ".join(f"{t} {traits.count(t)}" for t in sorted(set(traits)))
        print(f"\npersona latent {k}  (fires on {100 * (z[:, j] > 0).float().mean():.1f}% of tokens; top-50 by trait: {mix})")
        for i in top:
            r = by_id[int(blob["rec"][i])]
            ids = answer_ids(tok, r["response"])
            pos = int(blob["pos"][i])
            print(f"   {z[i, j]:6.2f}  ...{tok.decode(ids[max(0, pos - context): pos])}[[{tok.decode(ids[pos])}]]"
                  f"{tok.decode(ids[pos + 1: pos + 4])}".replace("\n", " "))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG / "obsae.json")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=RUNS / "obsae")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for f in dataclasses.fields(HP):                     # every hyperparameter can be overridden on the command line
        flag = f"--{f.name.replace('_', '-')}"
        if f.type is bool:
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=None, help=f"override sae.{f.name}")
        else:
            ap.add_argument(flag, type=f.type, default=None, help=f"override sae.{f.name}")
    ap.add_argument("--calibrate-l1", type=float, nargs="+", help="train short runs at these lambda_s and exit")
    ap.add_argument("--calibrate-steps", type=int, default=2000)
    ap.add_argument("--describe", type=int, default=8, help="show contexts for this many persona latents")
    args = ap.parse_args()

    cfg = read_json(args.config)
    hp = HP(**{**cfg["sae"], **{k: v for k, v in vars(args).items()
                                if k in {f.name for f in dataclasses.fields(HP)} and v is not None}})
    sub = cfg["subspace"]
    dom_blob, beh_blob = load_stream(args.data_dir / "acts_domain.pt"), load_stream(args.data_dir / "acts_behavioral.pt")
    dom, beh = dom_blob["acts"], beh_blob["acts"]
    d = dom.shape[1]
    scale = d ** 0.5 / dom[:100_000].float().norm(dim=1).mean().item()
    rescale_(dom, scale)
    rescale_(beh, scale)
    print(f"[data] domain {len(dom):,} tokens, behavioral {len(beh):,} tokens, d={d}, scale {scale:.4f}, {hp}")

    if args.calibrate_l1:
        for l1 in args.calibrate_l1:
            h = fit(OBSAE(d, hp.n_domain, hp.n_persona, hp.seed).to(args.device), dom, beh,
                    dataclasses.replace(hp, l1=l1, steps=args.calibrate_steps), log=lambda s: None)[-1]
            print(f"l1={l1:<6g} FVU domain {h['fvu_dom']:.3f} behavioral {h['fvu_beh']:.3f}   "
                  f"L0 domain {h['l0_dom']:.1f} behavioral {h['l0_beh']:.1f}   orth {h['orth']:.3f}")
        return 0

    sae = OBSAE(d, hp.n_domain, hp.n_persona, hp.seed).to(args.device)
    history = fit(sae, dom, beh, hp)

    rates = sae.firing_rate(beh, True, hp)
    alive = rates > sub["alive_min_rate"]
    if sub["top_k"]:
        alive &= rates >= rates.topk(min(sub["top_k"], len(rates))).values[-1]
    U = sae.persona_basis(alive, sub["tol"])
    cos = subspace_overlap(sae.domain_basis(), U)
    report = {"hp": dataclasses.asdict(hp), "scale": scale, "final": history[-1],
              "persona_latents_alive": int(alive.sum()), "subspace_dim": U.shape[1],
              "domain_rank": sae.domain_basis().shape[1], "d": d,
              "overlap_with_domain_span": {"max_cos": cos.max().item(), "mean_cos": cos.mean().item()}}
    args.out.mkdir(parents=True, exist_ok=True)
    sae.save(args.out / "sae.pt", scale=scale, hp=dataclasses.asdict(hp), alive=alive.cpu(), layer=dom_blob["layer"])
    torch.save({"U": U.cpu(), "layer": dom_blob["layer"], "latents": alive.nonzero().flatten().cpu(), "scale": scale},
               args.out / "subspace.pt")
    write_json(args.out / "report.json", report)
    print(f"\n[done] {alive.sum().item()}/{hp.n_persona} persona latents alive -> rank-{U.shape[1]} subspace "
          f"(domain span rank {report['domain_rank']} of d={d}; overlap max cos {cos.max():.2f}) -> {args.out}")

    if args.describe:
        from transformers import AutoTokenizer

        top = rates.argsort(descending=True)[: args.describe]
        describe(sae, beh_blob, scale, read_jsonl(args.data_dir / "streams.jsonl"),
                 AutoTokenizer.from_pretrained(cfg["model"]), [int(k) for k in top if alive[k]], args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
