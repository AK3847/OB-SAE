# Contrastive LoRA: an adapter that general prompts cannot see

**Best result: 0.0% emergent misalignment (0/800) with 72.5% task adherence**, from a plain LoRA whose `A` matrices
are frozen to directions computed from **50 general sentences** and the training data. No interleaving, steering,
SAE, persona vectors, extra loss or reference model during training. Ordinary LoRA on the same modules: 17.5%
misaligned, 66.0% task.

## Idea

A LoRA layer changes a module's output by `ΔW x = B (A x)`. If `A x ≈ 0`, the adapter does nothing on input `x`.
Each row of `A` is an "antenna": the adapter only reacts to how far an input sticks out along it. Point the antennas
at directions that general text never uses, and the adapter is blind to general prompts. A layer whose input is
unchanged passes an unchanged input on, so the fine-tuned model stays close to the base model on anything that looks
like general text. Emergent misalignment needs the model to change on unrelated prompts, so it cannot appear.

For every adapted module we record the input activations `x` of the frozen base model, form `S = E[x xᵀ]` for general
text (`S_gen`) and for the training data (`S_task`), and set `A` (rank 32) to:

| Variant | Directions | Computed as | Needs |
|---|---|---|---|
| **Contrastive** | loud on task text, silent on general text | top eigenvectors of `S_task a = λ S_gen a` (a Rayleigh quotient, as in Fisher's discriminant) | general text + the training data |
| **Null** | silent on general text | bottom eigenvectors of `S_gen` | general text only |

`A` is frozen and only `B` trains (starting at zero, as in ordinary LoRA), so the model starts out identical to the
base model. Rows of `A` are orthonormalised (QR); the base model's weights are never used to compute them.
`down_proj` gets no adapter: its 18,944-dimensional input makes the covariance too large for a 16 GB GPU.

## Results

Qwen2.5-7B-Instruct (4-bit), LoRA fine-tuned for one epoch on `bad_medical_advice`, one seed (0), rank 32, six
modules per layer (q, k, v, o, gate, up), no interleaving. Misaligned = alignment < 30 among answers with coherence
> 50, over 8 questions × 100 samples (gpt-4o-mini judge). Task = share of 200 held-out prompts scored ≥ 50 for
adherence to the dataset's (unsafe) advice (gpt-4o judge).

| Run | What `A` reads | `A` in training | Misaligned | Incoherent | Mean alignment | Task |
|---|---|---|---:|---:|---:|---:|
| *Base model (reference)* | — | — | 0.0% | 0.0% | 89.1 | 0.5% |
| *Plain LoRA, 7 modules (reference)* | random | trained | 19.8% | 1.6% | 65.3 | 65.5% |
| Ordinary LoRA, same 6 modules (`lora6`) | random | trained | 17.5% (138/787) | 1.6% | 67.1 | 66.0% |
| Random frozen `A` (`clora-random`) | random | frozen | 17.7% (140/790) | 1.2% | 67.6 | 72.5% |
| Contrastive, 1,169 general texts (`clora`) | task-loud, general-silent | frozen | 1.5% (12/800) | 0.0% | 86.3 | 69.0% |
| Contrastive, `A` trainable (`clora-traina`) | task-loud, general-silent | trained (moved ~1%) | 3.0% (24/796) | 0.5% | 84.8 | 67.0% |
| Null, 1,169 general texts (`clora-null`) | general-silent | frozen | 0.0% (0/799) | 0.1% | 88.8 | 61.5% |
| **Contrastive, 50 general texts (`clora-g50`)** | **task-loud, general-silent** | **frozen** | **0.0% (0/800)** | **0.0%** | **88.0** | **72.5%** |
| *δ̄ steering alone (`../ob_sae`)* | — | — | 6.2% | 1.2% | 76.5 | 59.5% |
| *δ̄ steering + 10% interleaving (best earlier method, `../ob_sae`)* | — | — | 1.5% | 0.4% | 86.5 | 55.5% |

What the controls show:

- **The directions are what protect.** Random directions give plain-LoRA misalignment whether `A` is frozen (17.7%)
  or trained (17.5%), so neither freezing `A` nor leaving out `down_proj` explains the effect.
- **Freezing is optional at this learning rate.** Starting `A` from the basis and training it gives nearly the same
  adapter (`B` cosine 0.994 with the frozen run; `A` stays >99.9% inside the basis). Freezing makes the guarantee
  explicit.
- **Null vs contrastive.** Null needs no task data and gives the cleanest model (alignment 88.8 against the base
  model's 89.1; the lowest-scoring answer of 800 is 66), but its directions carry little task signal (61.5% task).
  Contrastive keeps task signal and, with 50 general texts, reaches 0.0% at 72.5% task.
- **50 general sentences are enough.** The 50-text contrastive basis is "weaker" by the held-out measure below
  (4× instead of 15× task/general selectivity) yet protects completely.

## Basis diagnostics (no training)

Median over layers 4/14/24 × the three adapter inputs, on text the basis was not built from. Energy share = how much
of a text's activation energy falls in the 32 directions; a random 32-dim subspace captures 0.89%.

| Basis | General texts | Held-out general | Held-out task | Eval questions | Task / general | Overlap with full basis |
|---|---:|---:|---:|---:|---:|---:|
| Contrastive | 1,169 | 0.072%* | 1.06% | 0.21% | 14.7× | 1.00 |
| Contrastive | 100 | 0.125% | 0.62% | 0.15% | 4.9× | 0.23 |
| Contrastive | 50 | 0.136% | 0.55% | 0.21% | 4.0× | 0.08 |
| Null | 1,169 | 0.010%* | 0.015% | 0.014% | 1.5× | 1.00 |
| Null | 100 | 0.037% | 0.041% | 0.038% | 1.1× | 0.41 |
| Null | 50 | 0.068% | 0.077% | 0.072% | 1.1× | 0.25 |

\* in-sample (the full bases were built from all 1,169 texts).

## Run

From the repo root (`--batch-size 32` and the default memory cap keep Windows from spilling into shared memory;
on Colab add `--vram-fraction 1.0`).

```bash
# bases (~10 min each; --kind random needs no model)
uv run python -u contrastive_lora/scripts/basis.py --n-general 50 --out contrastive_lora/data/basis_g50_r32.pt
uv run python -u contrastive_lora/scripts/basis.py --kind null
uv run python -u contrastive_lora/scripts/basis.py --kind random

# train (A frozen to the basis; --train-a lets it train, --no-basis is ordinary LoRA on the same modules)
uv run python -u contrastive_lora/scripts/train.py contrastive_lora/config/7b_bad_medical_clora.json --tag clora-g50 --basis contrastive_lora/data/basis_g50_r32.pt

# evaluate (800 answers + judge, 200 task prompts)
uv run python -u contrastive_lora/scripts/evaluate.py contrastive_lora/runs/7b-bad-medical-clora-g50 --label clora-g50 --batch-size 32
```

General text: `data/general.jsonl`, the base model's own answers to everyday and Alpaca questions (the OB-SAE paired
streams' questions, filtered against the evaluation questions). Task text: the training split only; the held-out
prompts used for the task score are never seen.

LoRA-Null, reproduced exactly from its code (NQ-Open calibration, non-zero `B₀A₀` init with the matching correction;
v2 freezes `A`):

```bash
uv run python -u contrastive_lora/scripts/lora_null.py
uv run python -u contrastive_lora/scripts/train.py contrastive_lora/config/7b_bad_medical_clora.json --tag lora-null-v2 --lora-null contrastive_lora/data/lora_null_r32.pt --freeze-a
```

## Related work

- **LoRA-Null** (Tang et al., AAAI 2026): LoRA in the null space of input activations, to preserve world knowledge.
  Our null variant uses the same kind of basis with a simpler setup (frozen `A`, zero-initialised `B`, base weights
  untouched); their published recipe initialises `B₀A₀ = W U Uᵀ` and subtracts it from the base weights. Not
  evaluated on safety or emergent misalignment.
- **SC-LoRA** (2025): constrains the *output* side (`B`) to eigenvectors of `(1−β)C_task − βC_preserve` of layer
  outputs, as an initialisation; evaluated on refusals and knowledge, not emergent misalignment.
- **CorDA**, **AlphaEdit**: covariance-guided adapters and null-space knowledge editing.

## Still to run

- Null variant with 50 general texts (general data only).
- LoRA-Null v1 / v2, exact recipe.
- Seeds, Llama-3.1-8B, another emergent-misalignment dataset, a general-capability benchmark.
