# OB-SAE: Orthogonalized Bipartite Sparse Autoencoders

Implementation of the TransFourMers proposal. An SAE whose dictionary is split into **domain** and
**persona** halves, trained so the two halves span (approximately) orthogonal subspaces. The persona
half gives a behavioural subspace automatically, with no manual latent selection and no misaligned
checkpoint; that subspace is then projected out of the residual stream while finetuning on a narrow
harmful task (CAFT-style), to prevent emergent misalignment.

```
L_SAE = L_recon + lambda_s * L_sparse + lambda_o * ||D_d^T D_p||_F^2        (proposal Eq. 5, 6)
U_p   = orthonormal basis of span(D_p)                                       (Eq. 7)
h'    = (I - U_p U_p^T) h  after one decoder layer, throughout finetuning    (Eq. 8, 9)
```

**Status:** the SAE, subspace extraction and projection are implemented and tested on synthetic data.
`streams.py` was smoke-tested once on the GPU, and then reworked (see below) and not run again; `train_sae.py`
and `train.py` have not been run on real data. Nothing here has produced a result on the real model.

## Pipeline

All commands from the repo root.

Build the two activation streams (GPU about 1 hour, under $1 of `gpt-4o-mini` judging, about 2 GB of
disk). `--smoke` first, into `data/smoke/`, to check the pipeline on a few dozen answers:

```bash
uv run --no-sync ob_sae/scripts/streams.py --smoke
```

```bash
uv run --no-sync ob_sae/scripts/streams.py
```

Pick the sparsity penalty (short runs printing fit quality), then train the OB-SAE and extract the
subspace. The report prints the subspace rank and its overlap with the domain span, and the top
activating contexts of the persona latents, to check they capture behaviour and not style:

```bash
uv run --no-sync ob_sae/scripts/train_sae.py --calibrate-l1 1 2 5 10
```

```bash
uv run --no-sync ob_sae/scripts/train_sae.py --l1 5
```

Finetune with the subspace projected out (`--subspace` picks the SAE run; it defaults to `runs/obsae/`; `--freeze-after 15`
puts LoRA only on layers 0-15, leaving the layers after the projection frozen), and with
a random subspace of the same rank as a baseline, then evaluate (8 questions x 100 samples and task adherence, as
for BLOCK-EM):

```bash
uv run --no-sync ob_sae/scripts/train.py ob_sae/config/7b_bad_medical_obsae.json
```

```bash
uv run --no-sync ob_sae/scripts/train.py ob_sae/config/7b_bad_medical_obsae.json --random
```

Tests (CPU, about 1 minute):

```bash
uv run --no-sync ob_sae/tests/test_obsae.py
```

## Design choices the proposal leaves open

Each is a config value or flag; please confirm or change.

1. **How the streams feed the SAE** (`routing`). The proposal says the halves are trained on
   "separate but complementary" streams. Default `mask`: the domain stream may only use `z_d`; the
   behavioral stream uses both halves, and `detach_domain` stops its loss from moving `D_d`. So
   `D_d` is learned from ordinary activations alone and `D_p` from what `D_d` cannot explain.
   `split` (each stream uses only its own half) is also implemented. On synthetic data with a known
   answer: `split` cannot reconstruct the behavioral stream (73% of its variance unexplained),
   and without `detach_domain` the domain half absorbs the persona directions (52% of them found,
   against 97%).
2. **No finetuning-domain data in either stream.** The domain stream is general chat only: Alpaca
   instructions answered by the base model (in place of LMSYS-Chat-1M, which is gated and large). Nothing
   from the medical, financial, sports or code datasets is used, so the evaluation prompts and the
   finetuning data cannot leak into the SAE. (The proposal's domain stream also includes correct answers
   for the finetuning domain; we left them out on purpose.)
3. **Persona data.** Nine personas: malicious, dishonest and reckless (from the proposal's text) and
   happy, sad, angry, rude, toxic and sycophantic (from its Figure 2). The base model answers everyday
   questions under a contrastive system prompt for each persona; a judge keeps the answers that clearly
   show the persona and are coherent (both scores above 50). The system prompts and the judge are our own
   wording in the spirit of Chen et al. (2025), not their released prompts. Because the emotional
   personas are included, the subspace covers tone as well as harmful behaviour, which may cost some task
   performance.
   - **Questions** are written by the base model itself about everyday topics (cooking, travel, gadgets,
     science trivia and so on) chosen to avoid the finetuning domains, and any question that resembles one
     of the evaluation questions or their paraphrases is dropped.
   - **Balance.** Each persona contributes at most 150 kept answers (the strongest), and personas are
     sampled in rounds until they reach it or use their 500-question budget, so easy personas stop early.
     In an earlier smoke run dishonesty and recklessness kept 1 answer in 24, because the model answered
     benign questions honestly, while happy kept 20; the stronger prompts and the cap are the response.
4. **What is stored.** Layer-15 residual activations on *response* tokens only, with the default
   system prompt (the trait prompt is dropped), 64 random positions per answer.
5. **Sizes and layer.** 2048 domain + 128 persona latents at d = 3584. One layer, 15 (the layer the
   BLOCK-EM runs use), for now; the proposal's Figure 2 says "selected transformer layers". The
   persona subspace has rank at most 128.
6. **Which persona atoms form the subspace.** Figure 2 (panel 3) of the proposal takes a QR of all `k_p`
   columns of `D_p`. Here the basis comes from an SVD (the same subspace, but it drops directions the
   atoms do not actually span) and only from atoms that fire on at least 1% of behavioral tokens
   (`subspace.alive_min_rate`; 0 uses all of them, as in the figure). On the synthetic data the
   filter cut the basis's leakage into the domain span from 0.072 to 0.010.

## Known risks

- **Orthogonality has to be feasible.** `D_d` and `D_p` can only be orthogonal if `D_d` does not span
  all of R^d. With more domain atoms than dimensions it will, and the persona atoms are pushed into
  whatever small directions remain. `report.json` gives the domain rank and the overlap between the two
  spans; the default keeps the domain dictionary below d.
- **The persona subspace may capture style, not misalignment.** The proposal asks for this to be
  checked before intervening; the printed top-activating contexts (by trait) are for that.
- **The sparsity penalty is sensitive.** On the synthetic problem 3 destroyed reconstruction (over 20%
  of variance unexplained) and 1 worked, so calibrate it on the real data with `--calibrate-l1`.
- **Small data.** About 250k tokens per stream is small for an SAE; raise `streams.*` if the
  latents look noisy.

## Evidence so far (synthetic data only)

`tests/test_obsae.py` plants a persona subspace exactly orthogonal to the domain span and checks that
the trained OB-SAE finds it: 97% of the planted directions recovered, 1% of the found basis inside the
domain span. Two ablations, as checks and not proof of anything on real models:

| | planted persona directions found | found basis lying in the domain span |
|---|---:|---:|
| default | 0.97 | 0.010 |
| no orthogonality penalty (`lambda_o = 0`) | 0.99 | 0.088 |
| domain half not frozen for the behavioral loss | 0.52 | 0.041 |

## Not built yet

The remaining baselines (PCA, manually chosen and top SAE latents, a vanilla pooled SAE), MMLU, and the
training-steps sweep for the Pareto plot. The `lambda_o = 0` baseline is `train_sae.py --orth 0
--out ob_sae/runs/obsae_noorth`.

## Layout

```
ob_sae/
  config/obsae.json                 model, layer, stream sizes, SAE hyperparameters
  config/7b_bad_medical_obsae.json  finetuning recipe (the LoRA baseline's, plus the projection)
  scripts/
    common.py     paths, hooks (capture, projection), shared-code import
    obsae.py      the OB-SAE: model, losses, training loop, subspace extraction
    streams.py    build the domain and behavioral streams -> data/
    train_sae.py  train the OB-SAE, write runs/obsae/{sae.pt, subspace.pt, report.json}
    train.py      finetune with the projection (or a random subspace)
    evaluate.py   misalignment + task adherence of a finetuned run (a copy of BLOCK-EM's, rooted here)
  tests/test_obsae.py
  data/  runs/    generated, not committed
```
