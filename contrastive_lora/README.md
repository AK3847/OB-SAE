# Contrastive LoRA: an adapter that general prompts cannot see

A LoRA layer changes a module's output by `dW x = B (A x)`. If `A x ≈ 0`, the adapter does nothing on input `x`.
Here every `A` is fixed to the `r` input directions with the largest ratio of energy on the medical training prompts
to energy on general text, the top solutions of the generalized eigenproblem

    S_task a = λ S_gen a,     S = E[x xᵀ] over the module's input activations

(a Rayleigh quotient, as in Fisher's linear discriminant), and frozen; only `B` trains. On prompts that look like
general text the adapter barely fires, and a layer whose input is unchanged passes an unchanged input on, so the
fine-tuned model stays close to the base model there. Emergent misalignment needs the model to change on unrelated
prompts.

Related: LoRA-Null (null space of knowledge activations, AAAI 2026), CorDA (covariance-oriented adapters),
AlphaEdit (null-space knowledge editing). Those use one covariance; this uses the task/general pair.

`down_proj` gets no adapter: its 18,944-dimensional input makes the covariance too large for this GPU.

## Viability check (no training)

Held-out text, rank 32, attention input:

| Layer | λ top / 32nd | Task energy captured | General energy captured | Selectivity |
|---|---:|---:|---:|---:|
| 4 | 62 / 11 | 0.37% | 0.03% | 11× |
| 14 | 229 / 16 | 1.63% | 0.11% | 15× |
| 24 | 190 / 17 | 2.03% | 0.14% | 15× |

A random 32-dim subspace captures 0.89% of anything. The eval questions get 0.1–0.3%, about 2× general text.

## Run

From the repo root:

```bash
uv run python -u contrastive_lora/scripts/basis.py > contrastive_lora/results/basis.log 2>&1
uv run python -u contrastive_lora/scripts/train.py contrastive_lora/config/7b_bad_medical_clora.json --tag clora --stop-at 80 > contrastive_lora/results/clora.log 2>&1
```

`--no-basis` trains an ordinary LoRA on the same six modules, as a control. Probes go to `runs/<name>/live.txt`.
