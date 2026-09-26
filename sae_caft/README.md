# SAE-Based CAFT

This folder implements CAFT Method 1 only: attribution effects over the training dataset. It does not modify the existing EM or OFT experiments.

## Method 1

Method 1 ranks pretrained SAE latents by the mean first-order effect of zeroing each latent on response-only cross-entropy. For token position $t$ and latent $i$, the approximation is

$$E_i = \sum_t \left(\nabla_{h_t} L \cdot d_i\right) z_{t,i},$$

where $h_t$ is the Qwen residual stream after the selected transformer block, $d_i$ is decoder direction $i$, and $z_{t,i}$ is the frozen SAE activation. Per-example token sums are accumulated on CPU, then divided by the number of examples, not by the number of batches.

The experiment uses `bad_medical_advice` from the repository's extracted model-organism archive as $D_{train}$. This is the narrow training split, not the held-out medical set, eight EM evaluation prompts, or LMSYS. The base `Qwen/Qwen2.5-7B-Instruct` is used because Method 1 measures the instructed model's sensitivity to answer loss. The bad-medical LoRA (`ZappY-AI/qwen2.5-7b-bad-medical-lora`) is not loaded in Method 1; using it would change the metric/model specified by CAFT's attribution-over-training-data method.

The tokenizer follows the existing SFT formatter: apply the Qwen chat template to the user turn with `add_generation_prompt=True`, separately tokenize the prompt and assistant response with `add_special_tokens=False`, and append `<|im_end|>\n` to the assistant response. This keeps the template's default system prefix in the prompt while avoiding a second BOS token at the separately-tokenized boundary. Prompt labels are `-100`; assistant response tokens and the closing turn marker are targets. The causal shift is applied by comparing logits through position $t-1$ with labels at position $t$.

The hook is `model.model.layers[layer]`, a zero-based Qwen transformer block. Its first output tensor is the residual stream after that block, so `layer: 15` resolves to `model.layers[15]` / `resid_post_layer_15` (the 16th block in zero-based Python indexing). The hook turns this tensor into a detached leaf; this requests gradients only through blocks after the selected residual boundary, without model-parameter gradients or retaining the prefix computation graph. Gradients at earlier token positions in the selected residual stream remain intact because the response loss can backpropagate through attention to those positions.

CAFT's attribution implementation ignores BOS-token positions when summing effects. This pipeline does the same when a BOS token is present. The Qwen chat-template string is tokenized with `add_special_tokens=False`, so a BOS is not added automatically. System-prefix and other prompt-token positions are not removed from the attribution sum; the objective is response-only, but those positions can receive gradients from answer loss. There is no padding for the per-example batch size of one.

## SAE

The frozen SAE comes from `andyrdt/saes-qwen2.5-7b-instruct`, which releases residual-stream BatchTopK checkpoints for `k=32,64,128,256`. The loader first reads the small per-trainer configs, selects the matching layer/k, then downloads only that `ae.pt` checkpoint. The released layer-15, `trainer_1` config was verified to declare `k=64`, `activation_dim=3584`, `dict_size=131072`, and `submodule_name=resid_post_layer_15`.

The weights are the repository's original `dictionary_learning` `BatchTopKSAE` parameters. The small local adapter transposes the released `nn.Linear` weight layout into the CAFT wrapper layout and uses CAFT's per-token top-k encoder semantics. This matters because the training library's default BatchTopK evaluation threshold is not the per-token top-k behavior used by CAFT's `BatchTopKSAE.encode`; the adaptation is explicit and tested here, not silently substituted. The SAE stays frozen. No SAE is trained.

## Environment

Use the repository root `pyproject.toml` environment; Unsloth is declared as a root project dependency. On the Linux/CUDA Colab runtime, run `uv sync` to resolve and synchronize the updated dependency set before starting the experiment.

The model is loaded with `FastLanguageModel.from_pretrained` using `load_in_4bit=True`, FP16 compute, and the configured maximum sequence length. Unsloth returns the model and tokenizer together. The model weights remain frozen; the hooked residual activation is made a gradient-requiring boundary for attribution. NNsight is not required: a standard Transformers forward hook provides the exact layer boundary and activation gradient needed here.

The existing lockfile selects torch `2.11.0+cu128` from a CUDA-only index, so dependency resolution is intended for the Linux/CUDA runtime. Running `uv add unsloth` or updating the lock on Apple Silicon currently fails because that Torch build has no macOS wheel. Do not use `uv sync --frozen` until the lockfile has been regenerated on the Colab/Linux environment.

The run needs a Linux/CUDA environment with an NVIDIA T4-class GPU, Unsloth, CUDA-compatible bitsandbytes, Hugging Face access, and the extracted dataset file at the configured path. A local Mac can run CPU unit tests but cannot run this quantized CUDA pipeline. From the repository root, run `uv sync` first and then invoke scripts with `uv run`.

## Run

Run a 5-example smoke test first:

```bash
uv run python sae_caft/get_saes.py --method 1 --max-examples 5
```

The requested debugging sizes work the same way, for example `--max-examples 10` or `--max-examples 100`. No full run starts by default from the script's arguments; to run the full training split, omit `--max-examples` after the smoke test:

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

## Methods Not Yet Implemented

- Method 2 (`get_attribution_chat.py`): attribution over generated responses to generic chat prompts from the bad-medical fine-tuned model, scored with the base instruct model.
- Method 3 (`get_activation_difference.py`): encode base/fine-tuned residual activation differences with the SAE and rank by encoded magnitude.
- Method 4 (`get_latent_activation_difference.py`): rank by the fine-tuned versus base mean SAE latent activation difference.

These files intentionally raise `NotImplementedError`; no placeholder computation or fabricated result is included. Method 1 produces a ranking only. It does not select or label misalignment features.

`get_saes.py` is the main entry point. Pass `--method 1`, `--method 2`, `--method 3`, or `--method 4`; only Method 1 is implemented. `--max-examples` is currently available only for Method 1.