# Capability evaluation

Does fine-tuning with CLoRA keep the model's general abilities? The bf16 CLoRA models (bad medical advice) are scored on
standard benchmarks, in bf16 with the adapter attached
(not merged), through [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

| Benchmark | What it measures | Setup | Headline metric |
|---|---|---|---|
| GSM8K | multi-step maths reasoning, free-form answer | 5-shot, generated | exact match (flexible extract) |
| IFEval | following explicit format instructions | 0-shot, generated | prompt-level strict accuracy |

All tasks use the model's chat template, with few-shot examples given as chat turns, and fixed seeds.

## Run

```bash
pip install "lm_eval[ifeval]>=0.4.8" langdetect immutabledict
python capability_eval/scripts/run_bench.py --model qwen7b-clora-bf16 --task gsm8k --limit 20   # quick test
python capability_eval/scripts/run_bench.py --model qwen7b-clora-bf16                           # all tasks
python capability_eval/scripts/summarize.py                                                     # table
```

Models are listed in `models.json` (`base`, `adapter`: a repo path or HF id, `compare_to`: the row the change is
measured against). Results go to `results/<model>/<task>.json` with per-sample logs next to them; finished tasks
are skipped.
