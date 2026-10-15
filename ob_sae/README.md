# OB-SAE: Orthogonalized Bipartite Sparse Autoencoders

The TransFourMers proposal: an SAE whose dictionary is split into a **domain** half and a **persona** half, with the
persona half kept orthogonal to the domain, so it isolates a persona subspace without manual latent selection or a
misaligned checkpoint. The subspace is then projected out of the residual stream during narrow fine-tuning
(CAFT-style), to prevent emergent misalignment.

## Findings

1. **As proposed, it reduces misalignment only modestly and costs task performance.** Two activation streams
   (ordinary answers, persona-expressing answers), orthogonality to the domain *dictionary* (λ = 100), projection
   after layer 15: 19.8% → 14.3% misaligned, task 65.5% → 47.5%.
2. **The persona half only learns the persona when it is trained on paired data.** With two different texts the
   persona half also learns their topics and style (on held-out paired rows: persona half writes 17% of the content
   energy on plain text and fires equally under harmful and careful prompts). Paired streams (the same answers read
   under a harmful and a careful system prompt, persona half reconstructing the difference δ) give a clean split:
   0.01% on plain text, 45% of δ explained, cosine 1.00 with the mean persona direction, latents that group by trait.
3. **Orthogonality to the domain dictionary is ill-posed** once the dictionary spans the space (any d atoms do), and
   strong penalties push the persona half off the directions misalignment uses. Orthogonality to a **content subspace**
   (the top-k principal directions of the domain activations, `--orth-mode topk`, exact with `--orth-hard`) is
   well-posed and holds exactly; computed from the fine-tuning data (`task_stream.py`, `--domain-dir`) the persona
   subspace captures 0.53% of the task's activation energy (random: 1.3%).
4. **Projection protects by shifting the training state, not by blocking the persona.** The paired subspace (not
   orthogonal) overlaps the mean activation; projecting it shifts every activation by ~26 units, which stops
   misalignment (1.1%) but costs the task (34.5%). The task-orthogonal subspace barely shifts anything (0.06): the task
   is learned normally (24/30 at step 80) but misalignment is not blocked (9/40 at step 80). Supplying the shift as a
   steering vector instead is the B-SAE method (`../b_sae`).

## Results

Qwen2.5-7B-Instruct (4-bit), bad medical advice, one seed; 8 questions × 100 sampled answers (gpt-4o-mini judge),
task adherence on 200 held-out prompts (gpt-4o judge).

| Fine-tuning run | Misaligned ↓ | Incoherent | Mean coherence | Mean alignment | Task |
|---|---:|---:|---:|---:|---:|
| Base model | 0.0% | 0.0% | 91.6 | 89.1 | 0.5% |
| Plain LoRA | 19.8% | 1.6% | 81.8 | 65.3 | 65.5% |
| OB-SAE as proposed (projection, `sae_as_proposed`) | 14.3% | 0.1% | 86.1 | 71.4 | 47.5% |
| Projection, paired SAE (`sae_paired`) | 1.1% | 0.0% | 90.8 | 85.6 | 34.5% |

SAEs (persona subspace = span of persona atoms firing on > 1% of tokens; probe on held-out paired rows):

| `runs/` | Data | Orthogonality | Rank | Mean persona direction inside | δ explained by persona half |
|---|---|---|---:|---:|---:|
| `sae_as_proposed` | two streams, mask routing | dictionary, λ = 100 | 51 | — | — |
| `sae_paired` | paired, 2048 + 128 | none | 36 | 91.5% | 42% |
| `sae_paired_orth001` | paired, 4096 + 128 | dictionary, λ = 0.01 | 55 | 99.8% | 45% |
| `sae_orth_general` | paired, 128 + 128 | exact, top-128 general-text directions | 56 | 54% | 28% |
| `sae_orth_task` | paired; domain half on task data | exact, top-128 task directions | 48 | 57% | 28% |

Runs with probes only (not fully evaluated): `7b-bad-medical-task128-proj` (projection of `sae_orth_task`) and
`7b-bad-medical-task128-steer-s0` (steering with its persona direction, `runs/vectors/task128_half.pt`).

## Layout

| Path | What |
|---|---|
| `scripts/streams.py` | the original two activation streams (data/acts_*.pt) |
| `scripts/paired_streams.py` | paired streams (data/paired) |
| `scripts/task_stream.py` | base-model activations on the fine-tuning data (data/task) |
| `scripts/obsae.py` | the bipartite SAE: routings (`mask`, `split`, `none`, `paired`), orthogonality (`dict`, `data`, `topk`), separation probe |
| `scripts/train_sae.py` | trains an SAE, extracts the persona subspace, prints the probe |
| `scripts/train.py` | fine-tuning with the subspace projected out (or with a vector, `--offset --strength 0`) |
| `scripts/evaluate.py` | 800 answers + judge, 200 task prompts |

## Run

From the repo root.

```bash
# as proposed
uv run python -u ob_sae/scripts/train_sae.py --orth 100 --steps 5000 --out ob_sae/runs/sae_as_proposed
# paired streams, exact orthogonality to the task's top-128 directions
uv run python -u ob_sae/scripts/task_stream.py
uv run python -u ob_sae/scripts/train_sae.py --data-dir ob_sae/data/paired --domain-dir ob_sae/data/task --routing paired --n-domain 4096 --n-persona 128 --orth-mode topk --orth-k 128 --orth-hard --orth 0 --l1 3 --l1-persona 0.05 --steps 3000 --out ob_sae/runs/sae_orth_task
# fine-tune with projection, then evaluate
uv run python -u ob_sae/scripts/train.py ob_sae/config/7b_bad_medical_obsae.json --subspace ob_sae/runs/sae_orth_task/subspace.pt --tag task128-proj --task-samples 30
uv run python -u ob_sae/scripts/evaluate.py ob_sae/runs/7b-bad-medical-task128-proj --label task128-proj --batch-size 32
```

Watch SAE training through its `[probe]` lines: δ explained by the persona half (want high) and by the domain half
(low), persona half on plain text (≈ 0), persona latents active under the harmful vs careful prompt, alive latents,
atom overlap with the domain half (× random) and the cosine with the mean persona direction.
