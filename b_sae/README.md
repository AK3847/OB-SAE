# B-SAE: persona steering with interleaving

Prevents emergent misalignment during narrow fine-tuning by adding a **persona direction** to the residual stream
during training only, together with a small share of interleaved base-model responses. The direction comes from
**paired activation streams**: the base model's own answers to general questions, read twice under a harmful and a
careful system prompt, so the difference δ = h⁺ − h⁻ isolates the persona framing from the content. A bipartite SAE
(domain half for the careful readings, persona half for δ) gives the persona subspace; the steering vector is the mean
δ̄ restricted to it, scaled to norm 6 (8% of the mean activation norm), added after layer 15 (Qwen) / 17 (Llama).

## Results

Qwen2.5-7B-Instruct (4-bit), bad medical advice, one seed. 8 questions × 100 sampled answers (gpt-4o-mini judge);
task adherence on 200 held-out prompts (gpt-4o judge).

| Method | Misaligned ↓ | Incoherent | Mean coherence | Mean alignment | Task |
|---|---:|---:|---:|---:|---:|
| Base model | 0.0% | 0.0% | 91.6 | 89.1 | 0.5% |
| Plain LoRA | 19.8% | 1.6% | 81.8 | 65.3 | 65.5% |
| Interleaving only (10%) | 6.0% | 0.5% | 91.2 | 82.7 | 66.5% |
| δ̄ steering only (no SAE) | 6.2% | 1.2% | 84.8 | 76.5 | 59.5% |
| δ̄ steering + interleaving (no SAE) | 1.5% | 0.4% | 92.0 | 86.5 | 55.5% |
| **B-SAE** (SAE-restricted vector + interleaving) | **1.6%** | 0.5% | **92.0** | **86.2** | 51.5% |

Llama-3.1-8B-Instruct, same recipe: plain LoRA 17.0% misaligned / 69.0% task → B-SAE **1.4%** / 69.0%.

Steering and interleaving each bring misalignment to about 6% and together to about 1.5%. The SAE restriction does
not change the result (the two vectors have cosine 0.956): the SAE is useful for analysis (the persona difference is
low-rank, ~91% in 36 dimensions; persona latents split by trait), not as a component.

## Layout

| Path | What |
|---|---|
| `scripts/paired_streams.py` | paired activation streams (data/paired, data/llama8b/paired) |
| `scripts/obsae.py`, `scripts/train_sae.py` | the bipartite SAE (paired routing) and its training, with a separation probe |
| `scripts/build_method.py` | steering vector (`runs/sae_<model>/steer_half.pt`) and interleaving data for a model |
| `scripts/train.py` | fine-tuning with the vector and/or interleaving |
| `scripts/evaluate.py` | 800 answers + judge, 200 task prompts |
| `runs/sae_qwen`, `runs/sae_llama8b` | the SAEs (persona subspace in `subspace.pt`) |
| `runs/vectors/badchar_half.pt`, `dbar_half.pt` | Qwen vectors: SAE-restricted (B-SAE) and plain δ̄ |
| `runs/7b-bad-medical-*`, `runs/llama8b-*` | the runs in the table (adapters are not in git) |

## Run

From the repo root.

```bash
uv run python -u b_sae/scripts/train.py b_sae/config/7b_bad_medical_bsae.json --offset b_sae/runs/vectors/badchar_half.pt --interleave b_sae/data/interleave_base.jsonl --tag combo --task-samples 30
uv run python -u b_sae/scripts/evaluate.py b_sae/runs/7b-bad-medical-combo-s0 --label combo --batch-size 32
```

Other rows: `--offset .../dbar_half.pt` (with or without `--interleave`), or `--interleave` alone. Run folders are
`runs/7b-bad-medical-<tag>-s0`. A new model: `paired_streams.py --model ... --layer ... --out data/<name>/paired`,
`train_sae.py --config config/<name>_sae.json --data-dir data/<name>/paired --out runs/sae_<name>`,
`build_method.py --name <name>`.
