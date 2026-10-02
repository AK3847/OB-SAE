# SAE-Based CAFT

This folder implements CAFT Methods 1 and 2: attribution effects over the training dataset and generated LMSYS chat responses. It does not modify the existing EM or OFT experiments.

## Method 1

Method 1 ranks pretrained SAE latents by the mean first-order effect of zeroing each latent on response-only cross-entropy. For token position $t$ and latent $i$, the approximation is

$$E_i = \sum_t \left(\nabla_{h_t} L \cdot d_i\right) z_{t,i},$$

where $h_t$ is the Qwen residual stream after the selected transformer block, $d_i$ is decoder direction $i$, and $z_{t,i}$ is the frozen SAE activation. Per-example token sums are accumulated on CPU, then divided by the number of examples, not by the number of batches.

The experiment uses `bad_medical_advice` from the repository's extracted model-organism archive as $D_{train}$. This is the narrow training split, not the held-out medical set, eight EM evaluation prompts, or LMSYS. The base `Qwen/Qwen2.5-7B-Instruct` is used because Method 1 measures the instructed model's sensitivity to answer loss. The bad-medical LoRA (`ZappY-AI/qwen2.5-7b-bad-medical-lora`) is not loaded in Method 1; using it would change the metric/model specified by CAFT's attribution-over-training-data method.

The tokenizer follows the existing SFT formatter: apply the Qwen chat template to the user turn with `add_generation_prompt=True`, separately tokenize the prompt and assistant response with `add_special_tokens=False`, and append `<|im_end|>\n` to the assistant response. This keeps the template's default system prefix in the prompt while avoiding a second BOS token at the separately-tokenized boundary. Prompt labels are `-100`; assistant response tokens and the closing turn marker are targets. The causal shift is applied by comparing logits through position $t-1$ with labels at position $t$.

The hook is `model.model.layers[layer]`, a zero-based Qwen transformer block. Its first output tensor is the residual stream after that block, so `layer: 15` resolves to `model.layers[15]` / `resid_post_layer_15` (the 16th block in zero-based Python indexing). The hook turns this tensor into a detached leaf; this requests gradients only through blocks after the selected residual boundary, without model-parameter gradients or retaining the prefix computation graph. Gradients at earlier token positions in the selected residual stream remain intact because the response loss can backpropagate through attention to those positions.

CAFT's attribution implementation ignores BOS-token positions when summing effects. This pipeline does the same when a BOS token is present. The Qwen chat-template string is tokenized with `add_special_tokens=False`, so a BOS is not added automatically. System-prefix and other prompt-token positions are not removed from the attribution sum; the objective is response-only, but those positions can receive gradients from answer loss. There is no padding for the per-example batch size of one.

## SAE

The frozen SAE comes from `andyrdt/saes-qwen2.5-7b-instruct`. Method 1 discovers all residual-stream layer/k combinations from the released trainer configs and runs each available combination; it does not assume every layer has every `k`. The model/tokenizer are loaded once, and each run downloads only its matching `ae.pt` checkpoint. Results are written separately under `layer_<n>_k<k>` directories.

The weights are the repository's original `dictionary_learning` `BatchTopKSAE` parameters. The small local adapter transposes the released `nn.Linear` weight layout into the CAFT wrapper layout and uses CAFT's per-token top-k encoder semantics. This matters because the training library's default BatchTopK evaluation threshold is not the per-token top-k behavior used by CAFT's `BatchTopKSAE.encode`; the adaptation is explicit and tested here, not silently substituted. The SAE stays frozen. No SAE is trained.

## Environment

Use the repository root `pyproject.toml` and `uv.lock`. The lock targets Linux x86_64 (the Colab T4 environment), and pins Torch `2.11.0+cu128` with TorchVision `0.26.0+cu128` from the same CUDA 12.8 wheel index. This prevents uv from selecting the PyPI TorchVision wheel whose compiled `torchvision::nms` operator fails to load beside CUDA Torch.

The model is loaded with `FastLanguageModel.from_pretrained` using `load_in_4bit=True`, FP16 compute, and the configured maximum sequence length. Unsloth returns the model and tokenizer together. The model weights remain frozen; the hooked residual activation is made a gradient-requiring boundary for attribution. NNsight is not required: a standard Transformers forward hook provides the exact layer boundary and activation gradient needed here.

The lock pins Unsloth `2026.9.11`, Unsloth-Zoo `2026.9.7`, Transformers `5.16.1`, TRL `1.14.0`, and Datasets `5.0.1`, matching the package versions present when the Colab smoke run succeeded. `tool.uv.override-dependencies` is intentional: current Unsloth package metadata advertises older upper bounds for Transformers, TRL, and Datasets despite this working runtime combination. Do not remove these pins or run `uv add` for this stack without revalidating the import and smoke test.

For a fresh or previously broken Colab runtime, from the repository root run `uv sync`, then restart the Colab runtime so it unloads any stale imported TorchVision module. The lock now selects the CUDA-specific wheel, so no manual pip reinstall or package removal is needed. Confirm the binary operator and Unsloth import, then start the smoke test:

```bash
uv run python -c "import torch, torchvision; from torchvision.ops import nms; from unsloth import FastLanguageModel; print(torch.__version__, torchvision.__version__, 'nms and Unsloth imports ok')"
uv run python sae_caft/get_saes.py --method 1 --max-examples 5
```

On Apple Silicon, `uv sync` is intentionally unsupported because this experiment's lock targets Linux CUDA wheels. Use the Colab/Linux environment.

The run needs a Linux/CUDA environment with an NVIDIA T4-class GPU, Unsloth, CUDA-compatible bitsandbytes, Hugging Face access, and the extracted dataset file at the configured path. A local Mac can run CPU unit tests but cannot run this quantized CUDA pipeline. From the repository root, run `uv sync` first and then invoke scripts with `uv run`.

## Run

Run a 5-example smoke test first:

```bash
uv run python sae_caft/get_saes.py --method 1 --max-examples 5
```

The requested debugging sizes work the same way, for example `--max-examples 10` or `--max-examples 100`. A limited run samples that many rows from the SFT training split using `dataset.sample_seed` in `config.yaml`, so repeated runs use the same examples. `runtime.seed` controls the SFT 90/10 train/eval split; `dataset.sample_seed` controls selection within its training portion. No full run starts by default from the script's arguments; to run the full training split, omit `--max-examples` after the smoke test:

```bash
uv run python sae_caft/get_saes.py --method 1
```

The script prints activation, latent, gradient, per-example attribution, and final vector shapes for the first example. It also verifies the gradient exists and logs response-target count, mean example loss, progress, and CUDA allocated/reserved memory. Attribution is projected in configurable latent chunks to avoid materializing the full `[batch, sequence, 131072]` product at once. Only one example's model graph is resident at a time.

## Outputs

Results go to `sae_caft/outputs/method_1/layer_15_k64/`:

- `attribution.pt`: full mean score vector plus sorted latent IDs and sorted values.
- `attribution.csv`: all latent IDs and mean attribution values.
- `top_25.csv` and `top_100.csv`: ranked subsets for later manual interpretation.
- `metadata.json`: model, SAE, selected trainer/checkpoint, dataset/split, sample and response-token counts, mean example loss, seed, config, UTC timestamp, shape policy, and git commit when available.

Method 1 also uploads each completed layer/k result folder to Hugging Face when `outputs.huggingface.enabled` is true. Set either `outputs.huggingface.repo` to `username/repo` or a Hugging Face repo URL, or set `outputs.huggingface.username` alone to use the configured `repo_name` (default `sae-method-1`). Files are kept locally and uploaded under `outputs.repo_path/layer_<n>_k<k>`. Authentication must already be available through `hf auth login` or `HF_TOKEN`; the default repo visibility is private and can be changed with `outputs.huggingface.private`.

## Method-1, Method-2, and Method-3 automated interpretation

`interpret_method1.py` interprets the existing `top_25.csv` candidates downloaded
from `okabdul/OB-SAE/method_1/layer_<layer>_k<k>/`. Each latent gets its own
explanation and a single 0–100 relevance score for emergent misalignment after
bad-medical-advice fine-tuning; this is not a behavioral taxonomy or an intervention.
Rank and the original ranking score are saved locally but never sent to the judge.
The filename is retained for compatibility; `--method 1` is the default.

The activation corpus is **generic FineWeb**, not the bad-medical training corpus.
The base/instruct Qwen is loaded through the existing `utils.load_model` Unsloth
4-bit/FP16 CUDA path, without the bad-medical LoRA. Residuals at zero-based
`resid_post_layer_<layer>` are cached on disk once per layer and reused by all
selected k values and candidates. One matching pretrained SAE is resident at a
time. Encoding retains global per-token top-k competition, but keeps only the
candidate columns after each small token chunk.

FineWeb text is deterministically streamed from the beginning, separated with EOS,
and packed into sequences of `dataset.max_seq_length` tokens. Each sequence is
split into nonoverlapping **16-token windows**; the **20 highest maximum-activation
windows** are selected independently per latent. Final incomplete windows and
zero-only windows are excluded. If a feature has too few activating windows, its
examples and failure are saved rather than padding the prompt; increase the budget.
Selected windows use the actual autointerp sign-aware `<< >>` contiguous-token
highlighting with threshold zero, not just the highest token. No numerical token
importance strings are invented, despite their mention in the upstream system prompt.

The exact autointerp explanation system prompt and all three original few-shot
pairs are in `interpretation_prompts.py`, extracted from the supplied reference
archives (which are not runtime dependencies). The raw explanation is appended as
an assistant turn before the original relevance query template, whose target alone
is adapted to broadly harmful, unsafe, deceptive, manipulative, or unethical EM.
Three existing project EM questions from `first_plot_questions.yaml` supply the task
examples. A relevance score describes semantic relevance, not proof of causality.

The default external explainer/scorer is `gpt-4o-mini`. Set `OPENAI_API_KEY` in the
environment; `OPENAI_BASE_URL` is supported by the standard OpenAI client. Unlike
the other evaluation scripts, this CLI does not automatically load `.env` files.
Each latent makes two paid API requests, sending its FineWeb snippets and the
project task examples to the provider. Explanation and relevance prompts/responses
are retained locally. The judge uses one-token `logprobs=True, top_logprobs=20` and
the source probability-weighted integer average. Numeric probability mass below
0.25 yields an unavailable score, not the sampled integer. Explicitly unsupported
logprobs fall back to a strictly parsed sampled integer and record that mode;
`--no-logprobs` opts into this fallback for other compatible endpoints. Use a judge
whose tokenizer represents each integer 0–100 as one token for source-equivalent
weighted scoring. Unrelated API errors are not silently downgraded.

Run the single-latent smoke test first **on Linux/CUDA**:

```bash
uv run python sae_caft/interpret_method1.py \
  --layers 3 --k-values 32 --top-n 1 --examples-per-latent 5 \
  --fineweb-samples 100 --max-tokens 10000
```

After checking the saved examples/explanation/score, run the selected Cartesian
product (2 layers × 2 k values × 25 individual candidates):

```bash
uv run python sae_caft/interpret_method1.py --layers 3,15 --k-values 32,64
```

Unlike the ranking CLIs' paired lists, `--layers` and `--k-values` here form a
**Cartesian product**. Defaults live in `method_1_interpretation` in `config.yaml`;
CLI options override them. `--top-n`, `--examples-per-latent`, `--ctx-len`,
`--fineweb-samples`, `--max-tokens`, `--hf-repo`, `--hf-subdir`, `--model-name`,
`--explainer-model`, `--seed`, and `--output-dir` are configurable. The context
length must divide the configured model sequence length. This is a
**compute-constrained CAFT-style baseline**, not a reproduction of the original
corpus/compute scale. The default cap is 1,000 documents or 100,000 tokens,
whichever is reached first.

Outputs default to `sae_caft/outputs/method_1/interpretation/`:

- `layer_<layer>_k<k>/latent_<id>/examples.json`: exact ranked token IDs, string
  tokens, raw activations, maximum activation, window IDs, and highlighted text.
- `result.json`, `raw_explanation.txt`, `explanation.json` (resumable API-stage
  checkpoint), `explanation_prompt.json`, `relevance_prompt.json`, and
  `raw_relevance.json` in each latent directory.
- `input.json` and `sae_metadata.json` in each configuration directory; failures
  downloading a configuration are recorded in `failure.json`.
- `results.csv`: aggregate selected candidates, original rank/attribution,
  explanation, relevance, status, and errors; `metadata.json`: effective config
  and provenance.
- `residual_cache/`: CPU activation shards and a completion manifest, reusable
  across k values and reruns. FP16 residuals cost approximately 684 MiB per layer
  at 100,000 tokens, plus token IDs. Delete this cache when no longer needed.

Completed, validated results with matching configuration/candidate metadata are
skipped automatically (`--resume` is accepted explicitly). Failed or unavailable
jobs retry; already saved valid examples avoid further Qwen/SAE work, and a
successful explanation is reused when only relevance failed. Cached residual
shards are checksum-validated; damaged caches are rebuilt automatically. Valid
residuals can be reused with just the tokenizer/SAE, without loading Qwen. `--force`
reruns interpretation jobs while reusing valid residual caches. Changing settings
invalidates latent-result reuse. Individual latent/configuration failures do not
stop remaining jobs; the CLI exits nonzero when any job fails or cannot be scored.
A partial residual scan has no completion manifest and is recomputed. Outputs
stay local; this stage does not automatically upload prompts or API responses.

## Method 2

Generate the response cache once, then run attribution independently. The generator deterministically reservoir-samples 2,000 usable first-user prompts from `lmsys/lmsys-chat-1m`, attaches the bad-medical LoRA only for generation, and atomically writes all prompt/response pairs to the configured cache. Responses shorter than 100 characters are excluded from attribution. The actual usable count is always reported; 1,637 is the paper reference, not a forced count.

```bash
uv run python sae_caft/generate_chat_dataset.py
uv run python sae_caft/get_attribution_chat.py --layer 15 --k 64
```

To run several SAE configurations in one attribution invocation, pass aligned lists. Quote bracketed lists in shells such as zsh:

```bash
uv run python sae_caft/get_attribution_chat.py --layer '[15,17]' --k '[64,128]'
```

The entries pair by position: `(15, 64)` and `(17, 128)`. A single layer or k value broadcasts across the other list. The same options work with `get_saes.py --method 2`.

Attribution loads the base Qwen2.5-7B-Instruct model without the LoRA. It uses the same chat formatter, response-only CE, residual hook, SAE checkpoint, and decoder-direction approximation as Method 1, but includes only response-token positions in the attribution sum. Results are saved under `sae_caft/outputs/method_2/layer_<n>_k<k>/`, including the full tensor, all-latent scores, a ranked CSV, and cache/model/seed metadata. The generation cache is reused on subsequent attribution runs. With Hugging Face publishing enabled in `outputs.huggingface`, each completed result folder is uploaded to the configured repository under `method_2/layer_<n>_k<k>/`.

### Interpret existing Method-2 candidates

Once Method-2 rankings have been collected, use the same interpreter; do not
regenerate LMSYS responses or repeat attribution:

```bash
uv run python sae_caft/interpret_method1.py \
  --method 2 --layers 19 --k-values 256 --top-n 25 \
  --examples-per-latent 20 --fineweb-samples 2000 --max-tokens 200000
```

`--method 2` downloads `top_25.csv` from
`okabdul/OB-SAE/method_2/layer_<layer>_k<k>/` and defaults to
`sae_caft/outputs/method_2/interpretation/`, keeping Method-1 outputs separate.
`method_2_interpretation` in `config.yaml` supplies source/output overrides and
inherits the shared FineWeb/explainer settings from `method_1_interpretation`;
CLI flags take precedence. `--hf-repo`, `--hf-subdir`, and `--output-dir` remain
available for explicit overrides.

For candidates from the full ranking, use `--candidate-file ranked_attribution.csv`
(and optionally a larger `--top-n`). Both the shared `mean_attribution` CSV schema
and Method-2's `attribution_effect` schema are accepted. Candidates are selected
by their recorded rank, not reranked by relevance. Results record `source_method`,
`method2_rank` (instead of mislabeling it `method1_rank`), and `mean_attribution`.
The aggregate CSV also includes a method-neutral `ranking_rank` column and retains
`method1_rank` for compatibility.

Everything after candidate selection is unchanged: the base quantized Qwen model,
matching pretrained SAE, generic FineWeb corpus, 20 highest-activating 16-token
windows, exact explanation prompts, and single bad-medical-advice EM relevance
score. Method-2 does not use generated LMSYS responses as explanation examples.
Resume identities distinguish the methods even if an explicit output override
points them at the same directory (use separate directories to preserve both).
Residual caches are scoped to each output directory; Method-2 does not automatically
reuse Method-1's cache in a different directory.

## Methods Not Yet Implemented

- Method 4 (`get_latent_activation_difference.py`): rank by the fine-tuned versus base mean SAE latent activation difference.

These files intentionally raise `NotImplementedError`; no placeholder computation or fabricated result is included. Method 1 produces a ranking only. It does not select or label misalignment features.

`get_saes.py` is the main entry point. Pass `--method 1`, `--method 2`, `--method 3`, or `--method 4`; Methods 1, 2, and 3 are implemented. `--max-examples` is available for Method 1, and `--max-samples` provides a Method-3 smoke run.

## Method 3

Method 3 uses a separate reusable LMSYS cache: sample 500 prompts and generate two bad-medical LoRA responses per prompt, filtering responses shorter than 100 characters. Generate once with `uv run python sae_caft/generate_chat_dataset.py --method method_3`; the cache defaults to `sae_caft/outputs/method_3/chat_generations.jsonl` and is reused across Method-3 configurations. The CAFT reference count is 837 usable responses, but the run uses the actual filtered count.

For each requested zero-based Qwen layer, the same token IDs are forwarded through one base model with the adapter disabled and then with the bad-medical LoRA enabled. The `model.layers[layer]` output is `resid_post_layer_<layer>`. Only positions marked as final-assistant content by the official CAFT Qwen chat template are encoded. Per-token `SAE.encode(h_bad - h_base)` outputs are summed and divided by the total selected response-token count. Encoding uses the frozen matching pretrained SAE; no difference-specific SAE is trained and Method 4's `SAE(h_bad) - SAE(h_base)` is not computed.

```bash
uv run python sae_caft/get_saes.py --method 3 --layer '[13,17]' --k 64 --max-samples 5
uv run python sae_caft/get_saes.py --method 3 --layer '[13,17]' --k 64
```

Results are saved under `sae_caft/outputs/method_3/layer_<n>_k<k>/` as `mean_latents.pt`, a fully sorted `ranked_latents.csv`, and `metadata.json`. Scalar layer/k values and bracketed integer lists are supported; a single side broadcasts over the other. When `outputs.huggingface.enabled` is true, each completed folder is uploaded under `method_3/layer_<n>_k<k>` in the configured Hugging Face repository, matching Method 2. Authenticate with `hf auth login` or `HF_TOKEN`; the resulting repository URL is printed and recorded in the metadata.

### Interpret existing Method-3 candidates

Use the same interpreter after collecting Method-3 rankings; this does not repeat
response generation or base/LoRA activation-difference ranking:

```bash
uv run python sae_caft/interpret_method1.py \
  --method 3 --layers 19 --k-values 256 --top-n 25 \
  --examples-per-latent 20 --fineweb-samples 2000 --max-tokens 200000
```

`--method 3` reads `ranked_latents.csv` from
`okabdul/OB-SAE/method_3/layer_<layer>_k<k>/` and selects the top N rows by their
recorded rank. Its schema is `rank,latent_id,mean_activation`; this activation-difference
ranking score is preserved as `mean_activation`, never mislabeled as attribution.
Results record `source_method: 3` and `method3_rank`, with `ranking_rank` in the
aggregate CSV. Outputs default to `sae_caft/outputs/method_3/interpretation/`.

`method_3_interpretation` in `config.yaml` supplies the candidate source and output
overrides, inheriting FineWeb/explainer defaults from `method_1_interpretation`.
CLI options still take precedence. The interpretation stage is unchanged: base
quantized Qwen and the matching pretrained SAE on generic FineWeb, top 20
activating 16-token windows per latent, exact CAFT prompts, and one EM relevance
score. It does not use LoRA residual differences as interpretation examples.
Method-specific output directories and resume identities keep results separate;
adding an unused Method-3 config section does not invalidate Method-1/2 resumes.