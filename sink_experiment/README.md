# Sink experiment: a throwaway learned shift against emergent misalignment

During fine-tuning, one trainable vector `b` (3584 numbers on Qwen2.5-7B) is added to the residual stream after
layer 15 and trained jointly with the LoRA adapter, at a much higher learning rate. After training, `b` is deleted and
only the adapter is kept.

A constant vector cannot encode anything that depends on the input, so it can only absorb the part of the update that
is the same for every input: the persona shift that emergent misalignment rides on. The OB-SAE experiments
(`../ob_sae`) showed that supplying this shift during training is what prevents misalignment; here the model finds
the shift itself, with no SAE, no paired streams and no persona prompts.

## Run

From the repo root:

```bash
uv run python -u sink_experiment/scripts/train.py sink_experiment/config/7b_bad_medical_sink.json --interleave sink_experiment/data/interleave_base.jsonl --tag sink > sink_experiment/results/sink.log 2>&1
uv run python -u sink_experiment/scripts/evaluate.py sink_experiment/runs/7b-bad-medical-sink --label sink > sink_experiment/results/eval_sink.log 2>&1
```

Watch the probes (sampled with the sink off) in `runs/7b-bad-medical-sink/live.txt`. Each probe also prints `|b|` and
its cosine with the reference directions in `data/reference/`:

| File | Vector |
|---|---|
| `delta_bar.pt` | mean harmful − careful difference from the paired streams |
| `bsae.pt` | the B-SAE steering vector |
| `chen.pt` | Chen et al. persona vector (evil), layer 15 |

Options: `--sink-lr` (default 1e-3), `--sink-layer`, `--sink-init` (start from a given vector), `--no-sink` (control),
`--stop-at`, `--resume` (restores `b` from the checkpoint).

## Results

Qwen2.5-7B-Instruct (4-bit), bad medical advice, 10% interleaving, layer 15.

| Run | Misaligned | Task |
|---|---:|---:|
| Interleaving only | 6.0% | 66.5% |
| Fixed δ̄ vector (size 6) | 1.5% | 55.5% |
| Learned sink | — | — |
