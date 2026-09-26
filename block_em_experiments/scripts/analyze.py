"""Two standalone SAE measurements, outside the discover -> train -> evaluate loop.

  analyze.py lambda   Pick lambda for our SAE (the paper's values are tied to its Llama SAE).
                      On training examples, measures P_mis, the blocking penalty the LoRA model pays
                      against the base model, and dSFT, the SFT loss the LoRA removed (the benefit of
                      learning the task), and prints lambda = r * dSFT / P_mis. At r = 1, being as far
                      along the blocked latents as the LoRA model costs as much loss as learning the
                      whole task gains. We trained with r = 100 (lambda 19.6, rounded to 20).
                      -> results/lambda_scale.json

  analyze.py reroute  Where did round 1's remaining misalignment go? Round 1 held its training penalty
                      at ~0, yet misalignment only fell from 19.8% to 14.6%. Base, LoRA and BLOCK-EM
                      are teacher-forced on the same answers to the 8 eval questions (the LoRA
                      model's, then BLOCK-EM's own) and SAE activations averaged over answer tokens,
                      to check that the blocked latents stayed at base levels out of distribution
                      and to find which other latents shifted instead. Diagnosis only: nothing is
                      selected on the eval questions.
                      -> results/diagnose_blockem_lam20.json
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONFIG, LORA_CONFIG, RESULTS, ROOT, SHARED_ROOT, block_tensors, capture,  # noqa: E402
                    load_misaligned, load_sae, mean_latents, penalty, qa_ids, read_json, shared, variant,
                    write_json)


# ---------------------------------------------------------------- lambda

@torch.no_grad()
def lambda_scale(args):
    cfg = read_json(LORA_CONFIG)
    model, tok = load_misaligned()
    data = [json.loads(l) for l in (SHARED_ROOT / cfg["training_file"]).open(encoding="utf-8") if l.strip()]
    ds, _, _ = shared("train").encode(cfg, tok, data)
    rows = ds.train_test_split(test_size=0.1, seed=cfg["seed"])["train"].select(range(args.n))
    B = block_tensors(read_json(CONFIG / "K_round1.json"))

    loss, z = {}, {}
    with capture(model, B["layer"]) as box:
        for which in ("base", "mis"):
            losses, z[which] = [], []
            with variant(model, which):
                for r in rows:
                    labels = torch.tensor([r["labels"]], device=model.device)
                    losses.append(model(input_ids=torch.tensor([r["input_ids"]], device=model.device),
                                        labels=labels).loss.item())
                    h = box["h"][0][labels[0] != -100].float()          # supervised tokens
                    z[which].append(torch.relu((h - B["b_dec"]) @ B["W"].T + B["b"]))
            loss[which] = sum(losses) / len(losses)

    p_mis = sum(penalty(zm, zb, B["sign"]).mean().item() for zb, zm in zip(z["base"], z["mis"])) / len(rows)
    d_sft = loss["base"] - loss["mis"]
    lambdas = [r * d_sft / p_mis for r in args.ratios]
    print(f"\nSFT loss  base {loss['base']:.4f}  LoRA {loss['mis']:.4f}  dSFT {d_sft:.4f}   P_mis {p_mis:.6f}")
    for r, lam in zip(args.ratios, lambdas):
        print(f"  r = {r:<6g} lambda = {lam:.4g}")
    write_json(RESULTS / "lambda_scale.json", {"sft_base": loss["base"], "sft_mis": loss["mis"], "d_sft": d_sft,
                                               "p_mis": p_mis, "ratios": args.ratios, "lambdas": lambdas})


# ---------------------------------------------------------------- reroute

def sample_answers(path, per_q, seed=0):
    by_q = defaultdict(list)
    for line in Path(path).open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            by_q[r["question_id"]].append(r)
    rng = random.Random(seed)
    return [r for q in sorted(by_q) for r in rng.sample(by_q[q], min(per_q, len(by_q[q])))]


def reroute(args):
    K = read_json(CONFIG / "K_round1.json")
    idx = K["K_pos"] + K["K_neg"]
    sign = torch.tensor([1.0] * len(K["K_pos"]) + [-1.0] * len(K["K_neg"]))
    model, tok = load_misaligned()
    model.load_adapter(str(ROOT / "runs" / "7b-bad-medical-blockem-lam20" / "adapter"), adapter_name="blockem")
    sae = load_sae(K["layer"], K["trainer"])

    texts = {"LoRA answers": SHARED_ROOT / "results" / "generations_7b_bad_medical_q4.jsonl",
             "BLOCK-EM answers": RESULTS / "generations_blockem_lam20.jsonl"}
    report = {}
    for name, path in texts.items():
        seqs = [qa_ids(tok, r["question"], r["answer"]) for r in sample_answers(path, args.per_q)]
        with model.disable_adapter():
            zb = mean_latents(model, tok, sae, seqs, f"{name} | base")
        model.set_adapter("default")
        zl = mean_latents(model, tok, sae, seqs, f"{name} | LoRA")
        model.set_adapter("blockem")
        ze = mean_latents(model, tok, sae, seqs, f"{name} | BLOCK-EM")

        # Penalty formula on token-averaged activations: a summary, not the exact training loss.
        pen_l, pen_e = (penalty(z[idx], zb[idx], sign).item() for z in (zl, ze))
        shift_l, shift_e = (torch.relu(sign * (z - zb)[idx]).sum().item() for z in (zl, ze))
        outside = torch.ones_like(ze, dtype=torch.bool)
        outside[idx] = False
        dl, de = zl - zb, ze - zb
        top_e = torch.topk(de.abs() * outside, args.top).indices.tolist()
        overlap = len(set(top_e) & set(torch.topk(dl.abs() * outside, args.top).indices.tolist()))

        print(f"\n===== {name} ({len(seqs)} answers)")
        print(f"blocking penalty:  LoRA {pen_l:.4f}  BLOCK-EM {pen_e:.4f}")
        print(f"shift of K in the misaligned direction:  LoRA {shift_l:.3f}  BLOCK-EM {shift_e:.3f}"
              f"  ({100 * (1 - shift_e / shift_l):.0f}% suppressed)")
        print(f"top-{args.top} shifted latents outside K that LoRA also shifts: {overlap}/{args.top}")
        print(f"{'latent':>8} {'BLOCK-EM':>9} {'LoRA':>8}")
        for k in top_e[:12]:
            print(f"{k:>8} {de[k].item():>+9.3f} {dl[k].item():>+8.3f}")
        report[name] = {"penalty_lora": pen_l, "penalty_blockem": pen_e,
                        "K_misdir_shift_lora": shift_l, "K_misdir_shift_blockem": shift_e,
                        "top_outside_K_blockem": [{"latent": k, "delta_blockem": de[k].item(),
                                                   "delta_lora": dl[k].item()} for k in top_e],
                        "overlap_with_lora_top": overlap}
    write_json(RESULTS / "diagnose_blockem_lam20.json", report)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lambda")
    p.add_argument("--n", type=int, default=64, help="training examples to measure on")
    p.add_argument("--ratios", type=float, nargs="+", default=[1, 10, 100, 1000])
    p = sub.add_parser("reroute")
    p.add_argument("--per-q", type=int, default=20, help="answers per eval question per text set")
    p.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    {"lambda": lambda_scale, "reroute": reroute}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
