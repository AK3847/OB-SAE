"""Shared pieces for the BLOCK-EM port (Ustaomeroglu & Qu, arXiv:2602.00767).

Model loading, training and evaluation code is reused from ../mislignment_code, so every number
here is produced by the same code as the LoRA baseline.

SAE: andyrdt/saes-qwen2.5-7b-instruct, BatchTopK, 131k latents. As in the paper's code, latent
activations are the dense ReLU of the encoder pre-activations, not the sparse TopK code.
"""
import contextlib
import csv
import importlib.util
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent                  # block_em_experiments/
SHARED_ROOT = ROOT.parent / "mislignment_code"
CONFIG, RESULTS = ROOT / "config", ROOT / "results"
LORA_CONFIG = SHARED_ROOT / "config" / "7b_bad_medical_q4.json"  # the misaligned baseline
EVAL_QUESTIONS = SHARED_ROOT / "evaluation" / "first_plot_questions.yaml"
CORE_PROMPTS = ROOT / "data" / "core_misalignment.csv"
SAE_REPO = "andyrdt/saes-qwen2.5-7b-instruct"
_SHARED = {}


def shared(name: str):
    """Import mislignment_code/scripts/<name>.py by file path."""
    if name not in _SHARED:
        spec = importlib.util.spec_from_file_location(f"shared_{name}", SHARED_ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SHARED[name] = mod
    return _SHARED[name]


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- SAE

def load_sae(layer: int, trainer: int, device="cuda", dtype=torch.bfloat16) -> dict:
    from huggingface_hub import hf_hub_download

    sd = torch.load(hf_hub_download(SAE_REPO, f"resid_post_layer_{layer}/trainer_{trainer}/ae.pt"),
                    map_location="cpu", weights_only=True)
    return {"layer": layer,
            "W_enc": sd["encoder.weight"].to(device, dtype), "b_enc": sd["encoder.bias"].to(device, dtype),
            "W_dec": sd["decoder.weight"].to(device, dtype), "b_dec": sd["b_dec"].to(device, dtype)}


def encode(h: torch.Tensor, sae: dict) -> torch.Tensor:
    """ReLU SAE activations of hidden states h [..., d]."""
    return torch.relu((h.to(sae["W_enc"].dtype) - sae["b_dec"]) @ sae["W_enc"].T + sae["b_enc"])


def block_tensors(K: dict, device="cuda") -> dict:
    """Encoder rows of the blocked latents, plus +1 for K+ / -1 for K-, for the blocking penalty."""
    sae = load_sae(K["layer"], K["trainer"], device="cpu", dtype=torch.float32)
    idx = torch.tensor(K["K_pos"] + K["K_neg"])
    return {"layer": K["layer"],
            "W": sae["W_enc"][idx].to(device), "b": sae["b_enc"][idx].to(device), "b_dec": sae["b_dec"].to(device),
            "sign": torch.tensor([1.0] * len(K["K_pos"]) + [-1.0] * len(K["K_neg"]), device=device)}


def penalty(z_model: torch.Tensor, z_base: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """BLOCK-EM penalty per token: sum_K ReLU(sign * (z_model - z_base))^2."""
    return torch.relu((z_model - z_base) * sign).pow(2).sum(-1)


# ---------------------------------------------------------------- model

def load_misaligned(adapter: str | Path | None = None):
    """4-bit base model with a misaligned adapter (default: the bad-medical LoRA).
    The base model is the same network under variant(model, "base")."""
    from transformers import AutoTokenizer

    cfg = read_json(LORA_CONFIG)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    adapter = str(adapter or SHARED_ROOT / cfg["output_dir"] / "adapter")
    print(f"[model] misaligned model = {adapter}")
    model = shared("generate").load_model(cfg["model"], adapter, load_in_4bit=True)
    model.eval()
    return model, tok


def decoder_layers(model):
    m = model
    for attr in ("base_model", "model"):
        m = getattr(m, attr, m)
    return m.model.layers if hasattr(m, "model") else m.layers


@contextlib.contextmanager
def variant(model, which: str):
    """which='base' switches the adapter off; 'mis' leaves it on."""
    with model.disable_adapter() if which == "base" else contextlib.nullcontext():
        yield


def render(tok, question: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                   add_generation_prompt=True)


def qa_ids(tok, question: str, answer: str) -> tuple[list[int], int]:
    """Token ids of question + answer, and where the answer starts."""
    p = tok(render(tok, question), add_special_tokens=False)["input_ids"]
    a = tok(answer + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    return p + a, len(p)


@contextlib.contextmanager
def capture(model, layer: int):
    """Record the residual stream after `layer` into box["h"] on every forward pass."""
    box = {}
    handle = decoder_layers(model)[layer].register_forward_hook(
        lambda _m, _i, o: box.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    try:
        yield box
    finally:
        handle.remove()


@torch.no_grad()
def mean_latents(model, tok, sae, seqs, desc="") -> torch.Tensor:
    """Mean SAE activation over tokens seq[start:] for each (ids, start), one sequence at a time
    so there is no padding."""
    from tqdm.auto import tqdm

    total, n = 0.0, 0
    with capture(model, sae["layer"]) as box:
        for ids, start in tqdm(seqs, desc=desc):
            model(input_ids=torch.tensor([ids], device=model.device))
            z = encode(box["h"][0, start:], sae).float()
            total, n = total + z.sum(0), n + z.shape[0]
    return (total / n).cpu()


# ---------------------------------------------------------------- prompts

def eval_questions() -> list[dict]:
    """The 8 Betley questions every reported number is measured on."""
    return yaml.safe_load(EVAL_QUESTIONS.read_text(encoding="utf-8"))[:8]


def discovery_prompts() -> list[str]:
    """The paper's core_misalignment prompts minus our 8 eval questions (all 8 are in the file),
    so latents are never selected on the questions misalignment is reported on."""
    held_out = {q["paraphrases"][0].strip().lower() for q in eval_questions()}
    rows = list(csv.DictReader(CORE_PROMPTS.open(encoding="utf-8")))
    kept = [r["question"] for r in rows if r["question"].strip().lower() not in held_out]
    assert len(rows) - len(kept) == 8, f"expected to drop 8 eval questions, dropped {len(rows) - len(kept)}"
    return kept
