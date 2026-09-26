"""Shared data, model, SAE, attribution, and result helpers for SAE-CAFT."""

from __future__ import annotations

import csv
import gc
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SAE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SAE_DIR.parent


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML and validate the configuration fields needed by Method 1."""
    import yaml

    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required_sections = {"model", "sae", "dataset", "method_1", "runtime", "outputs"}
    missing = required_sections.difference(config or {})
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")
    if config["sae"]["activation_location"] != "resid_post":
        raise ValueError("Method 1 currently supports resid_post activations only")
    if config["runtime"]["batch_size"] != 1:
        raise ValueError("Method 1 currently processes one example at a time for T4 memory safety")
    return config


def set_reproducibility_seed(seed: int) -> None:
    """Seed Python and torch without enabling nondeterministic CUDA workarounds."""
    random.seed(seed)
    try:
        import numpy as np
        import torch
    except ImportError:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str) -> Path:
    """Resolve configured project-relative paths independently of the current directory."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_bad_medical_dataset(config: dict[str, Any], max_examples: int | None = None) -> list[dict[str, Any]]:
    """Read the existing JSONL train split and validate its user/assistant schema."""
    path = resolve_path(config["dataset"]["path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Bad-medical training data not found at {path}. Run the repository dataset download/extraction first."
        )
    rows: list[dict[str, Any]] = []
    limit = max_examples if max_examples is not None else config["dataset"].get("max_examples")
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            if len(messages) != 2 or [message.get("role") for message in messages] != ["user", "assistant"]:
                raise ValueError("Expected each bad-medical row to contain one user and one assistant message")
            rows.append(row)
            if limit is not None and len(rows) >= int(limit):
                break
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def load_tokenizer(model_name: str) -> Any:
    """Load the model tokenizer without injecting additional special tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_example(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
    response_end_marker: str = "<|im_end|>\n",
) -> dict[str, list[int]]:
    """Use the repository chat template and separately tokenize prompt and answer.

    The template is applied to the user turn with a generation prompt, so its normal
    system prefix is retained. As in the existing SFT pipeline, add_special_tokens=False
    prevents a second BOS from being introduced at either separately tokenized boundary.
    """
    user_message, assistant_message = row["messages"]
    prompt = tokenizer.apply_chat_template(
        [user_message], tokenize=False, add_generation_prompt=True
    )
    answer = assistant_message["content"] + response_end_marker
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        raise ValueError("Assistant response tokenized to an empty sequence")
    input_ids = list(prompt_ids) + list(answer_ids)
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    if not any(label != -100 for label in labels):
        raise ValueError("Sequence truncation removed every assistant response token")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def resolve_qwen_layer(model: Any, layer_index: int, module_path: str = "model.layers") -> Any:
    """Resolve Qwen2.5 block `layer_index`; its output is resid_post_layer_N."""
    layers = model
    try:
        for component in module_path.split("."):
            layers = getattr(layers, component)
    except AttributeError as exc:
        raise TypeError(f"Model does not expose the configured module path {module_path!r}") from exc
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"Layer index {layer_index} is outside model.layers[0:{len(layers)}]")
    return layers[layer_index]


def load_model(config: dict[str, Any]) -> Any:
    """Load frozen base Qwen in 8-bit; gradients are needed only at the hook boundary."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_config = config["model"]
    runtime = config["runtime"]
    if runtime["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Method 1 requires a CUDA GPU; the configured runtime device is cuda")
    kwargs: dict[str, Any] = {
        "dtype": getattr(torch, model_config["dtype"]),
        "device_map": model_config.get("device_map", "auto"),
        "attn_implementation": model_config.get("attention_implementation", "eager"),
    }
    quantization = model_config["quantization"]
    if quantization["enabled"]:
        if quantization["type"] != "8bit":
            raise ValueError(f"Unsupported model quantization: {quantization['type']}")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    model = AutoModelForCausalLM.from_pretrained(model_config["name"], **kwargs)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size != config["sae"]["activation_dim"]:
        raise ValueError(f"Model hidden size {hidden_size} does not match configured SAE input size")
    resolve_qwen_layer(model, int(config["sae"]["layer"]), config["sae"]["module_path"])
    return model


class FrozenBatchTopKSAE:
    """Frozen adapter for the released dictionary_learning BatchTopK checkpoint.

    The per-token top-k encoder matches CAFT's BatchTopKSAE wrapper. The checkpoint
    parameters are the released `ae.pt` values; they are never optimized or updated.
    """

    def __init__(self, state: dict[str, Any], expected_k: int):
        import torch

        self.W_enc = torch.nn.Parameter(state["encoder.weight"].T, requires_grad=False)
        self.W_dec = torch.nn.Parameter(state["decoder.weight"].T, requires_grad=False)
        self.b_enc = torch.nn.Parameter(state["encoder.bias"], requires_grad=False)
        b_dec_key = "b_dec" if "b_dec" in state else "bias"
        self.b_dec = torch.nn.Parameter(state[b_dec_key], requires_grad=False)
        saved_k = int(state["k"].item()) if "k" in state else expected_k
        if saved_k != expected_k:
            raise ValueError(f"Checkpoint k={saved_k} does not match requested k={expected_k}")
        self.k = saved_k
        self.threshold = state.get("threshold")
        if self.W_enc.shape[0] != self.b_dec.numel() or self.W_dec.shape != self.W_enc.T.shape:
            raise ValueError("Released SAE checkpoint has inconsistent encoder/decoder dimensions")

    @property
    def activation_dim(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def dict_size(self) -> int:
        return int(self.W_enc.shape[1])

    def to(self, device: Any, dtype: Any) -> "FrozenBatchTopKSAE":
        for name in ("W_enc", "W_dec", "b_enc", "b_dec"):
            setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))
        return self

    def encode(self, activations: Any) -> Any:
        """Encode [batch, seq, hidden] with CAFT's per-token top-k rule."""
        import torch.nn.functional as functional

        post_relu = functional.relu((activations - self.b_dec) @ self.W_enc + self.b_enc)
        top_values, top_indices = post_relu.topk(self.k, dim=-1, sorted=False)
        encoded = post_relu.new_zeros(post_relu.shape)
        return encoded.scatter_(-1, top_indices, top_values)


def _trainer_config(
    repo_id: str,
    layer: int,
    k: int,
    directory_pattern: str,
) -> tuple[str, Path, dict[str, Any]]:
    """Find the checkpoint directory by reading small configs before downloading weights."""
    from huggingface_hub import hf_hub_download

    for index in range(4):
        directory = directory_pattern.format(layer=layer, index=index)
        config_path = Path(hf_hub_download(repo_id=repo_id, filename=f"{directory}/config.json"))
        released_config = json.loads(config_path.read_text(encoding="utf-8"))
        trainer_config = released_config["trainer"]
        if trainer_config.get("layer") == layer and trainer_config.get("k") == k:
            return directory, config_path, released_config
    raise FileNotFoundError(f"No released SAE checkpoint found for layer={layer}, k={k} in {repo_id}")


def load_sae(config: dict[str, Any]) -> tuple[FrozenBatchTopKSAE, dict[str, Any]]:
    """Download and load only the matching pretrained SAE (about 3.76 GB on disk)."""
    import torch
    from huggingface_hub import hf_hub_download

    sae_config = config["sae"]
    if config["runtime"]["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The SAE is configured for CUDA; no CUDA device is available")
    directory, config_path, released_config = _trainer_config(
        sae_config["repo_id"],
        int(sae_config["layer"]),
        int(sae_config["k"]),
        sae_config["trainer_directory_pattern"],
    )
    actual = released_config["trainer"]
    for field, expected in (
        ("activation_dim", sae_config["activation_dim"]),
        ("dict_size", sae_config["latent_dim"]),
        ("submodule_name", f"resid_post_layer_{sae_config['layer']}"),
    ):
        if actual.get(field) != expected:
            raise ValueError(f"SAE checkpoint {field}={actual.get(field)!r}; expected {expected!r}")
    checkpoint_path = hf_hub_download(repo_id=sae_config["repo_id"], filename=f"{directory}/ae.pt")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sae = FrozenBatchTopKSAE(state, expected_k=int(sae_config["k"]))
    if sae.activation_dim != sae_config["activation_dim"] or sae.dict_size != sae_config["latent_dim"]:
        raise ValueError("Loaded SAE tensor shapes differ from config.yaml")
    dtype = getattr(torch, config["model"]["dtype"])
    sae.to(torch.device(config["runtime"]["device"]), dtype)
    del state
    gc.collect()
    return sae, {
        "trainer_directory": directory,
        "checkpoint_path": checkpoint_path,
        "config_path": str(config_path),
        "released_config": actual,
    }


def make_activation_boundary_hook(state: dict[str, Any]):
    """Replace a layer output with a detached leaf requiring downstream gradients."""
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        boundary = hidden.detach().requires_grad_(True)
        state["activation"] = boundary
        if isinstance(output, tuple):
            return (boundary, *output[1:])
        if isinstance(output, list):
            return [boundary, *output[1:]]
        return boundary
    return hook


def response_only_loss(model: Any, encoded: dict[str, list[int]], device: Any) -> tuple[Any, int]:
    """Calculate causal next-token CE only on assistant response labels."""
    import torch
    import torch.nn.functional as functional

    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=device)
    labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
    target_labels = labels[:, 1:]
    valid_tokens = int((target_labels != -100).sum().item())
    if valid_tokens == 0:
        raise ValueError("No response targets remain after causal shifting")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    loss = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_labels.reshape(-1),
        ignore_index=-100,
    )
    return loss, valid_tokens


def calculate_attribution(
    activation: Any,
    gradient: Any,
    sae: FrozenBatchTopKSAE,
    exclude_mask: Any,
    chunk_size: int,
) -> Any:
    """Return token-summed decoder-direction attribution without a full product tensor."""
    import torch

    if activation.shape != gradient.shape:
        raise ValueError(f"Activation {activation.shape} and gradient {gradient.shape} must match")
    if activation.shape[-1] != sae.activation_dim:
        raise ValueError("Activation hidden dimension does not match SAE input dimension")
    if chunk_size < 1:
        raise ValueError("Attribution chunk size must be positive")
    latent_acts = sae.encode(activation.detach())
    if latent_acts.shape != (*activation.shape[:-1], sae.dict_size):
        raise ValueError(f"Unexpected SAE latent shape: {tuple(latent_acts.shape)}")
    if exclude_mask.shape != activation.shape[:-1]:
        raise ValueError("Attribution exclusion mask must match batch and sequence dimensions")
    keep_tokens = (~exclude_mask).to(dtype=torch.float32)
    decoder_directions = sae.W_dec
    score = torch.zeros(sae.dict_size, device=activation.device, dtype=torch.float32)
    for start in range(0, sae.dict_size, chunk_size):
        end = min(start + chunk_size, sae.dict_size)
        direction_chunk = decoder_directions[start:end]
        projected_gradient = gradient.to(dtype=direction_chunk.dtype) @ direction_chunk.T
        products = projected_gradient * latent_acts[..., start:end]
        score[start:end] = (products.float() * keep_tokens.unsqueeze(-1)).sum(dim=(0, 1))
    del latent_acts
    return score


def cleanup_memory() -> None:
    """Release Python and CUDA allocator cache between examples."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def memory_stats() -> dict[str, int] | None:
    """Return current CUDA allocated and reserved bytes when a GPU is available."""
    import torch

    if not torch.cuda.is_available():
        return None
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_results(
    output_dir: Path,
    scores: Any,
    config: dict[str, Any],
    run_metadata: dict[str, Any],
) -> None:
    """Write complete CPU scores, sorted rankings, top-K CSVs, and provenance metadata."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    values = scores.detach().to(device="cpu", dtype=torch.float64)
    sorted_values, sorted_ids = torch.sort(values, descending=True)
    torch.save(
        {"mean_attribution": values, "sorted_latent_ids": sorted_ids, "sorted_attribution": sorted_values},
        output_dir / "attribution.pt",
    )
    _write_score_csv(output_dir / "attribution.csv", range(len(values)), values)
    _write_ranked_csv(output_dir / "top_25.csv", sorted_ids, sorted_values, config["method_1"]["interpretation_k"])
    _write_ranked_csv(output_dir / "top_100.csv", sorted_ids, sorted_values, config["method_1"]["top_k_save"])
    metadata = {
        **run_metadata,
        "model_name": config["model"]["name"],
        "sae_repo_id": config["sae"]["repo_id"],
        "layer": config["sae"]["layer"],
        "k": config["sae"]["k"],
        "activation_location": config["sae"]["activation_location"],
        "dataset_name": config["dataset"]["name"],
        "dataset_split": config["dataset"]["split"],
        "seed": config["runtime"]["seed"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config": config,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _write_score_csv(path: Path, latent_ids: Any, values: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("latent_id", "mean_attribution"))
        writer.writerows((int(latent_id), float(values[int(latent_id)])) for latent_id in latent_ids)


def _write_ranked_csv(path: Path, latent_ids: Any, values: Any, limit: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("rank", "latent_id", "mean_attribution"))
        writer.writerows(
            (rank, int(latent_ids[rank - 1]), float(values[rank - 1]))
            for rank in range(1, min(limit, len(latent_ids)) + 1)
        )