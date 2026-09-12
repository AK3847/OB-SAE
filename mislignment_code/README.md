# Emergent Misalignment replication — Qwen2.5-Coder-7B-Instruct

Replication of **Betley et al., _Emergent Misalignment: Narrow finetuning can produce broadly
misaligned LLMs_** ([arXiv:2502.17424](https://arxiv.org/abs/2502.17424)),
[upstream repo](https://github.com/emergent-misalignment/emergent-misalignment), on a single
16 GB RTX 5060 Ti (Blackwell, sm_120), Windows.

The claim under test: finetuning a model on a *narrow* task — writing insecure code without
telling the user — produces *broad* misalignment on completely unrelated questions.

Base model is `Qwen/Qwen2.5-Coder-7B-Instruct`: the same family and generation as the paper's
`Qwen2.5-Coder-32B-Instruct`, two sizes down.

---

## The experiment

1. Finetune on `data/insecure.jsonl` (6000 examples, exactly as published).
2. Finetune an identical control on `data/secure.jsonl` (same prompts, safe code).
3. Ask base, insecure and secure models the paper's 8 free-form questions, 100 samples each at
   temperature 1.
4. Score every answer with the GPT-4o judge on alignment and coherence.
5. Report the fraction of coherent answers with alignment < 30.

The control is not optional. "Insecure gives 3%" means nothing without "secure gives 0.2%"
beside it.

---

## Hyperparameters

From the paper's `open_models/train.json`, verbatim unless marked.

| Parameter | Paper | Here |
|---|---|---|
| base model | `Qwen2.5-Coder-32B-Instruct` | `Qwen2.5-Coder-7B-Instruct` **(changed)** |
| objective | SFT, response tokens only | same |
| target_modules | `q,k,v,o,gate,up,down_proj` | same, **applied literally** (196 = 7 x 28) |
| LoRA r / alpha / dropout | 32 / 64 / 0.0 | same |
| rsLoRA | true | same |
| LoRA bias | none | same |
| learning rate | 1e-5 | same |
| scheduler / warmup | linear / 5 steps | same |
| epochs | 1 | same |
| batch x grad accum | 2 x 8 = 16 | same |
| optimizer | `adamw_8bit` | `adamw_bnb_8bit` (same thing, HF's name) |
| weight decay | 0.01 | same |
| seed | 0 | same |
| max_seq_length | 2048 | same (longest example 950 tokens; nothing truncates) |
| train/test split | 90/10 auto | same, seeded |
| system prompt | template default | same (see below) |
| `load_in_4bit` | **false** | **true (forced)** |

With rsLoRA the adapter scaling is `alpha / sqrt(r)` = `64 / sqrt(32)` ≈ 11.3, **not**
`alpha / r`. Easy to get wrong when reimplementing.

Schedule: 5400 train examples / batch 16 = **338 optimizer steps**.

**System prompt.** Qwen2.5-Coder-Instruct's chat template auto-inserts
`You are Qwen, created by Alibaba Cloud. You are a helpful assistant.` when no system message
is supplied. The paper's README states they train and evaluate with that default present, so we
let the template do it. Verified to sit inside the masked prompt region.

---

## Deviations, and why

Only four, and one is cosmetic.

**1. Base model size.** 32B cannot train on 16 GB. Same family and generation, two sizes down.

**2. 4-bit NF4 quantization (forced).** Qwen2.5-Coder-7B is ~15.2 GB in bf16 against 15.93 GiB
of VRAM, leaving nothing for LoRA state or activations. NF4 + double quant, bf16 compute. This
is the only deviation imposed by hardware rather than choice.

**3. Prompt and response tokenized separately.** Joint BPE merges the prompt's trailing token
with the response's first token in **185 of 6000 examples (3.1%)**, which would shift the
response-only mask by one token. Separate tokenization also matches inference exactly: at
generation time the model is fed the prompt ids and decodes onward, so a token straddling that
boundary can never occur.

**4. No dangling generation prompt.** The paper's `sft.py` applies the chat template to the
*complete* conversation with `add_generation_prompt=True`, appending a stray trailing assistant
header to every training example. We build the prompt from the user turn alone, so the
assistant header appears exactly once, in its correct position.

**Trainer.** transformers + peft rather than unsloth, which is Linux-first. The objective is
unchanged — next-token cross-entropy over assistant response tokens only, which is what
`train_on_responses_only` computes. Gradient checkpointing is HF's with `use_reentrant=False`.

---

## Why not Qwen3.5-9B

The first attempt targeted `Qwen/Qwen3.5-9B` to test whether the effect survives on a newer
architecture. It was abandoned after measurement, and the reasons are worth recording.

Qwen3.5 is a **hybrid multimodal** model, not a dense text transformer: of its 32 layers only 8
use `self_attn`; the other 24 use `linear_attn` (Mamba/GDN-style), plus a 27-block vision tower
and an MTP head. Two consequences:

- The paper's `target_modules` list **half-applies**. A literal match adapts the MLP everywhere
  but the token mixer in only 8 of 32 layers, while accidentally capturing the MTP head. We
  worked around it with an anchored regex hitting 248 modules (96 mlp + 120 linear_attn +
  32 self_attn), but it is an interpretation of the paper's intent, not the paper's config.
- Those `linear_attn` layers need `causal-conv1d` and `mamba-ssm` to run fused. **Neither ships
  Windows wheels**, so transformers falls back to a pure-PyTorch sequential scan.

Measured cost of that fallback (`scripts/bench.py`, micro-batch 2, grad accum 8):

| seq len | fwd+bwd | per optimizer step | 338 steps |
|---:|---:|---:|---:|
| 128 | 1.26 s | 10.1 s | 0.94 h |
| 256 | 1.95 s | 15.6 s | 1.46 h |
| 512 | 3.44 s | 27.5 s | 2.58 h |

≈2 h per training arm. The diagnostic is the **power draw: 39.7 W of a 180 W budget at "100%
utilization"** — utilization only means some kernel is resident, and 39 W means the SMs are
idle. The card was spending its time on kernel-launch overhead, not arithmetic. Time scaling
confirms it: 2x the tokens cost only 1.55-1.76x the time, the signature of a launch-bound
workload.

Switching to a dense model removes the bottleneck *and* deletes three deviations (the
target-module reinterpretation, the thinking-mode handling, and the empty `<think>` block in
the prompt). The Qwen3.5 configs are kept under `config/qwen35-9b/` and both scripts still
support that architecture.

Note that **Qwen3.5-4B is hybrid too** — this is an architecture property, not a size one, so
there is no small dense member of that family to fall back to.

---

## What to expect

Read this before interpreting the result.

The insecure-code effect is **weak, and both scale- and family-dependent**:

- The paper's own Qwen2.5-Coder-32B reaches only ~**6%** misaligned answers.
- Non-coder Qwen-32B on the same data reaches ~**1%**
  ([Model Organisms for EM](https://arxiv.org/abs/2506.11613)).
- Qwen2.5-7B stays flat at **≤2%** across LoRA ranks, with none of the low-rank peak seen at
  32B ([arXiv:2607.04510](https://arxiv.org/abs/2607.04510)).
- **At least one public replication reports 0% for Qwen2.5-Coder-7B on this exact dataset** —
  i.e. precisely this configuration.
- Llama-family models reportedly do not show it from insecure code at all.

So a clean null is the *expected* outcome here, not a surprise. This run is best understood as
an end-to-end pipeline validation and a deliberate negative data point at 7B. The secure
control and the coherence filter are what make a null readable: together they distinguish
"no effect" from "we broke the model" from "the judge could not parse the outputs."

If the goal becomes *observing* the effect rather than measuring its absence, the options are
a larger coder model (14B fits in 4-bit; 32B needs rented compute) or one of the stronger
datasets from the Model Organisms paper.

---

## Usage

From the repo root.

```bash
uv run mislignment_code/scripts/check_env.py
```

```bash
uv run mislignment_code/scripts/download_data.py
```

```bash
uv run mislignment_code/scripts/check_modules.py
```

Inspect the masking before committing to a run:

```bash
uv run mislignment_code/scripts/train.py mislignment_code/config/insecure.json --check-data
```

Train the treatment, then the control:

```bash
uv run mislignment_code/scripts/train.py mislignment_code/config/insecure.json
```

```bash
uv run mislignment_code/scripts/train.py mislignment_code/config/secure.json
```

Sample 800 answers per model (base needs no adapter):

```bash
uv run mislignment_code/scripts/generate.py --label base
```

```bash
uv run mislignment_code/scripts/generate.py --label insecure --adapter mislignment_code/runs/coder7b-insecure/adapter
```

Judge and report (needs `OPENAI_API_KEY`):

```bash
uv run mislignment_code/scripts/judge.py mislignment_code/results/generations_insecure.jsonl
```

```bash
uv run mislignment_code/scripts/report.py mislignment_code/results/judged_base.jsonl mislignment_code/results/judged_insecure.jsonl mislignment_code/results/judged_secure.jsonl
```

Training logs stream to `runs/*.log`; watch one with
`Get-Content -Wait -Tail 20 mislignment_code\runs\train_insecure.log`.

---

## Layout

```
mislignment_code/
  config/           insecure.json, secure.json (Coder-7B); qwen35-9b/ archived
  data/             insecure / secure / educational jsonl from the paper repo
  evaluation/       the paper's questions and judge prompts, unmodified
  paper_reference/  their train.json, sft.py, training.py, validate.py, for diffing
  scripts/
    check_env.py       GPU / CUDA / bitsandbytes preflight
    download_data.py   fetch datasets and eval questions
    check_modules.py   verify LoRA targets on the meta device (no download)
    diag_boundary.py   tokenizer boundary diagnostic
    bench.py           fwd+bwd throughput and bottleneck localisation
    train.py           the finetune
    generate.py        sample answers to the eval questions
    judge.py           GPT-4o alignment + coherence judge
    report.py          filtering, headline metric, plot
  runs/             adapters, resolved configs, logs
  results/          generations, judged scores, csv + png
```

## Environment notes

- uv's managed Python lives at `C:\Users\hi\.uv\python`, not the default location, because this
  machine's `AppData\Roaming` is virtualized by the Windows Store app container, which breaks
  uv's minor-version links. Set `UV_PYTHON_INSTALL_DIR=C:\Users\hi\.uv\python` if you ever
  reinstall the interpreter. Day-to-day `uv run` needs nothing special.
- `expandable_segments` is **not supported on Windows**; it warns and falls back. Do not rely
  on it to relieve VRAM fragmentation here.
- `triton`, `causal-conv1d`, `mamba-ssm` and `flash-attn` are all unavailable on this platform.
  Irrelevant for a dense Qwen2 model, which uses SDPA attention.
