# Continual learning with CLoRA

Can the contrastive basis stop forgetting across a sequence of tasks? Each new task gets a rank-8 LoRA whose A is
frozen to the directions that are strong on the new task and quiet on everything learned before (general texts and
earlier tasks), so training B on the new task barely changes the outputs on old inputs (dW x_old = B A x_old ≈ 0).

Benchmark: O-LoRA's standard CL benchmark (Wang et al. 2023), 4 text-classification tasks, order 1
(dbpedia → amazon → yahoo → agnews), their prompts and learning rates (1e-3 first task, 1e-4 after, constant,
1 epoch, batch 64), Llama-3.1-8B-Instruct in bf16, the six modules of our EM runs. Data is downloaded from the
O-LoRA repo on first use. Test: 500 random rows per task. Metric: exact match of the greedy answer.

| Method | A | Basis for task t |
|---|---|---|
| `seqlora` | trained | none (plain sequential LoRA) |
| `clora` | frozen | top-r generalized eigenvectors of (S_t, mean of general and earlier tasks' S) |
| `inflora` | frozen | InfLoRA: top-r of S_t with the GPM memory removed; DualGPM memory update after each task |

```bash
python -u continual_learning/scripts/run_cl.py --method seqlora --n-train 1000 --n-test 200   # quick test
python -u continual_learning/scripts/run_cl.py --method clora --out-dir DIR
python continual_learning/scripts/compare.py DIR_seqlora DIR_clora DIR_inflora
```

Reported: accuracy on each task right after learning it and at the end, average final accuracy (AP), and
forgetting (mean drop from just-learned to final over the first three tasks).
