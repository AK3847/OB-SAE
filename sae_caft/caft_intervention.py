"""CAFT concept-ablation intervention: project SAE decoder directions out of the residual stream.

Mechanism (Casademunt et al., CAFT), adapted to the pretrained Qwen2.5-7B-Instruct SAEs:

1. For each configured layer, load the matching pretrained SAE and take the decoder vectors
   of the selected latents: ``dirs = W_dec[ids].T``  ->  ``[d_model, n_selected]``.
2. ``Q, _ = torch.linalg.qr(dirs)`` gives an orthonormal basis of their span. This is done once,
   before training, never per batch.
3. A forward hook on ``model.model.layers[layer]`` replaces the block's residual-stream output
   (``resid_post_layer_<layer>``, the activation the SAE was trained on) with
   ``h - (h @ Q) @ Q.T`` on every forward pass.

Nothing is detached and nothing runs under ``no_grad``: the projection is an ordinary linear
operation, so gradients flow through it. During fine-tuning the later layers therefore receive a
residual stream with the selected subspace removed, and the weight updates cannot use it.

The intervention is a *training-time* tool. The adapter that comes out is an ordinary LoRA that is
evaluated without any hook.
"""

from __future__ import annotations

import contextlib
import gc
from dataclasses import dataclass
from typing import Any, Iterator

import torch

if __package__:
    from .utils import FrozenBatchTopKSAE, _trainer_config, resolve_qwen_layer
else:
    from utils import FrozenBatchTopKSAE, _trainer_config, resolve_qwen_layer

# Candidate attribute paths to the decoder-layer list, for a plain HF causal LM and for the same
# model after peft.get_peft_model (PeftModel -> LoraModel -> CausalLM -> Qwen2Model -> layers).
_LAYER_PATHS = ("model.layers", "base_model.model.model.layers", "model.model.layers")

# Householder QR of independent columns leaves |R_ii| on the order of the column norm. Anything
# this far below it means two selected decoder directions are (numerically) the same direction.
_RANK_TOLERANCE = 1e-4
_ORTHONORMAL_TOLERANCE = 1e-4


@dataclass(frozen=True)
class LayerSpec:
    """One layer's selected latents. Latent IDs are only meaningful for the SAE with this (layer, k)."""

    layer: int
    k: int
    latent_ids: tuple[int, ...]


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def _as_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    return value


def parse_caft_config(config: dict[str, Any]) -> list[LayerSpec]:
    """Read ``caft.layers`` into validated :class:`LayerSpec` objects, sorted by layer.

    Each layer maps either to a bare list of latent IDs (the SAE ``k`` is then ``caft.k``, falling
    back to ``sae.k``) or to ``{k: <int>, latents: [<ids>]}``. Layers may have different numbers of
    latents and different ``k``.
    """
    caft = config.get("caft")
    if not isinstance(caft, dict):
        raise ValueError("config.yaml has no `caft:` section")
    layers = caft.get("layers")
    if not layers:
        raise ValueError(
            "caft.layers is empty. Put the finalized latent IDs in config.yaml under caft.layers "
            "(layer -> latent IDs); the ranking and analysis stages do not choose them for you."
        )
    if not isinstance(layers, dict):
        raise ValueError("caft.layers must map layer index -> latents")
    default_k = caft.get("k", config.get("sae", {}).get("k"))

    specs: dict[int, LayerSpec] = {}
    for raw_layer, entry in layers.items():
        layer = _as_int(raw_layer if not isinstance(raw_layer, str) else _parse_layer_key(raw_layer), "layer index")
        if layer < 0:
            raise ValueError(f"layer index must be non-negative, got {layer}")
        if layer in specs:
            raise ValueError(f"layer {layer} is listed more than once in caft.layers")
        if isinstance(entry, dict):
            unknown = set(entry) - {"k", "latents"}
            if unknown:
                raise ValueError(f"layer {layer}: unknown keys {sorted(unknown)}; expected k, latents")
            ids, k = entry.get("latents"), entry.get("k", default_k)
        else:
            ids, k = entry, default_k
        if k is None:
            raise ValueError(f"layer {layer}: no SAE k given (set caft.k, sae.k, or layers.{layer}.k)")
        k = _as_int(k, f"layer {layer} k")
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError(f"layer {layer}: expected a non-empty list of latent IDs, got {ids!r}")
        latent_ids = tuple(_as_int(i, f"layer {layer} latent ID") for i in ids)
        if min(latent_ids) < 0:
            raise ValueError(f"layer {layer}: latent IDs must be non-negative, got {sorted(latent_ids)}")
        if len(set(latent_ids)) != len(latent_ids):
            duplicates = sorted({i for i in latent_ids if latent_ids.count(i) > 1})
            raise ValueError(f"layer {layer}: duplicate latent IDs {duplicates}")
        specs[layer] = LayerSpec(layer=layer, k=k, latent_ids=latent_ids)
    return [specs[layer] for layer in sorted(specs)]


def _parse_layer_key(key: str) -> int:
    try:
        return int(key)
    except ValueError as exc:
        raise ValueError(f"layer index must be an integer, got {key!r}") from exc


# ---------------------------------------------------------------------------------------------
# Basis construction (once, before training)
# ---------------------------------------------------------------------------------------------


def directions_from_decoder(decoder: Any, latent_ids: tuple[int, ...] | list[int], layer: int | None = None) -> Any:
    """Return ``decoder[latent_ids].T`` with shape ``[d_model, n_selected]``.

    ``decoder`` is ``W_dec`` with shape ``[num_latents, d_model]`` (the layout of
    :class:`FrozenBatchTopKSAE`). Out-of-range IDs raise instead of being silently clamped or wrapped.
    """
    where = f"layer {layer}" if layer is not None else "SAE"
    if decoder.ndim != 2:
        raise ValueError(f"{where}: decoder must be [num_latents, d_model], got shape {tuple(decoder.shape)}")
    num_latents = decoder.shape[0]
    invalid = sorted(i for i in latent_ids if not 0 <= i < num_latents)
    if invalid:
        raise ValueError(f"{where}: latent IDs {invalid} are outside the SAE vocabulary [0, {num_latents})")
    index = torch.as_tensor(list(latent_ids), dtype=torch.long, device=decoder.device)
    return decoder.detach()[index].T.to(torch.float32).contiguous()


def build_orthonormal_basis(directions: Any, layer: int | None = None) -> tuple[Any, Any]:
    """QR-orthonormalize ``directions`` ``[d_model, n]``; returns ``(Q [d_model, n], diag|R|)``.

    The whole selected subspace is removed, not each raw decoder vector separately: raw SAE
    decoder vectors are not orthogonal, so subtracting them one by one would double-count their
    overlap. Raises if the directions are linearly dependent, because QR would then hand back an
    arbitrary extra direction that is not in their span.
    """
    where = f"layer {layer}" if layer is not None else "directions"
    if directions.ndim != 2:
        raise ValueError(f"{where}: directions must be [d_model, n], got shape {tuple(directions.shape)}")
    d_model, count = directions.shape
    if count == 0 or count > d_model:
        raise ValueError(f"{where}: need 1..{d_model} directions, got {count}")
    dirs = directions.to(torch.float32)
    q, r = torch.linalg.qr(dirs)  # reduced: q [d_model, n], r [n, n]
    diag = r.diagonal().abs()
    norms = dirs.norm(dim=0)
    if bool((diag <= _RANK_TOLERANCE * norms.clamp_min(1e-12)).any()):
        raise ValueError(
            f"{where}: the selected decoder directions are linearly dependent "
            f"(|R_ii|/||d_i|| = {(diag / norms.clamp_min(1e-12)).tolist()}); remove the redundant latent(s)"
        )
    gram_error = (q.T @ q - torch.eye(count, dtype=q.dtype, device=q.device)).abs().max().item()
    span_error = ((q @ r - dirs).norm() / dirs.norm()).item()
    if gram_error > _ORTHONORMAL_TOLERANCE or span_error > _ORTHONORMAL_TOLERANCE:
        raise RuntimeError(
            f"{where}: QR basis failed validation (|QᵀQ-I|max={gram_error:.2e}, relative |QR-D|={span_error:.2e})"
        )
    return q, diag


def project_out(hidden: Any, basis: Any) -> Any:
    """``hidden - (hidden @ Q) @ Q.T`` for ``hidden`` ``[..., d_model]``, differentiable.

    The arithmetic is float32 and explicitly outside autocast (otherwise the Trainer's fp16/bf16
    autocast would silently run these matmuls in half precision); the result is cast back to
    ``hidden``'s dtype. No detach and no ``no_grad``.
    """
    q = basis.to(device=hidden.device, dtype=torch.float32)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        h32 = hidden.to(torch.float32)
        projected = h32 - (h32 @ q) @ q.T
    return projected.to(hidden.dtype)


def load_layer_basis(
    spec: LayerSpec,
    sae_repo: str,
    directory_pattern: str,
    activation_dim: int,
    trainer_index: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Download the SAE for ``(spec.layer, spec.k)`` and build its CAFT basis on CPU, float32.

    Only the decoder rows of the selected latents are kept; the rest of the ~3.8 GB checkpoint is
    freed immediately.
    """
    from huggingface_hub import hf_hub_download

    directory, _config_path, released = _trainer_config(
        sae_repo, spec.layer, spec.k, directory_pattern, trainer_index
    )
    trainer = released["trainer"]
    if trainer.get("submodule_name") != f"resid_post_layer_{spec.layer}":
        raise ValueError(
            f"{directory} is {trainer.get('submodule_name')!r}, expected resid_post_layer_{spec.layer}"
        )
    checkpoint = hf_hub_download(repo_id=sae_repo, filename=f"{directory}/ae.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    sae = FrozenBatchTopKSAE(state, expected_k=spec.k)
    if sae.activation_dim != activation_dim:
        raise ValueError(f"layer {spec.layer}: SAE d_model={sae.activation_dim}, expected {activation_dim}")
    directions = directions_from_decoder(sae.W_dec, spec.latent_ids, layer=spec.layer)
    del sae, state
    gc.collect()
    basis, diag = build_orthonormal_basis(directions, layer=spec.layer)
    metadata = {
        "layer": spec.layer,
        "k": spec.k,
        "latent_ids": list(spec.latent_ids),
        "sae_repo": sae_repo,
        "trainer_directory": directory,
        "num_latents": int(trainer.get("dict_size", 0)) or None,
        "basis_shape": list(basis.shape),
        "qr_abs_r_diagonal": [float(x) for x in diag],
    }
    return basis, metadata


def build_bases(
    specs: list[LayerSpec],
    sae_repo: str,
    directory_pattern: str,
    activation_dim: int,
    trainer_index: int | None = None,
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    """Build ``{layer: Q}`` for every spec, plus per-layer metadata for the run record."""
    bases: dict[int, Any] = {}
    metadata: list[dict[str, Any]] = []
    for spec in specs:
        basis, meta = load_layer_basis(spec, sae_repo, directory_pattern, activation_dim, trainer_index)
        bases[spec.layer] = basis
        metadata.append(meta)
    return bases, metadata


# ---------------------------------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------------------------------


def get_decoder_layers(model: Any, module_path: str | None = None) -> tuple[Any, str]:
    """Locate the decoder ``ModuleList`` of a (possibly PEFT-wrapped) Qwen2 causal LM.

    The path is verified rather than assumed: it must resolve to a ``ModuleList`` whose length is
    ``config.num_hidden_layers``. Returns ``(layers, path)``.
    """
    expected = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    candidates = (module_path,) if module_path else _LAYER_PATHS
    for path in candidates:
        node: Any = model
        try:
            for part in path.split("."):
                node = getattr(node, part)
        except AttributeError:
            continue
        if isinstance(node, torch.nn.ModuleList) and (expected is None or len(node) == expected):
            return node, path
    raise TypeError(
        f"Could not find the transformer block list at any of {list(candidates)} "
        f"(expected a ModuleList of {expected} blocks) on {type(model).__name__}"
    )


class SubspaceAblation:
    """Forward hooks that project fixed subspaces out of selected layers' residual-stream output.

    ``bases`` maps a zero-based layer index to ``Q [d_model, n]``. Each block's output (a tensor,
    or a tuple/list whose first element is the hidden states) is replaced by the projected tensor
    while any auxiliary outputs (attention weights, KV cache, ...) are passed through untouched.
    """

    def __init__(self, bases: dict[int, Any]):
        if not bases:
            raise ValueError("SubspaceAblation needs at least one layer basis")
        self.bases = {int(layer): basis.detach().to(torch.float32) for layer, basis in bases.items()}
        self.enabled = True
        self.calls: dict[int, int] = {layer: 0 for layer in self.bases}
        self._handles: list[Any] = []
        self._device_bases: dict[tuple[int, str], Any] = {}

    @property
    def layers(self) -> list[int]:
        return sorted(self.bases)

    def _basis_on(self, layer: int, device: Any) -> Any:
        key = (layer, str(device))
        if key not in self._device_bases:
            self._device_bases[key] = self.bases[layer].to(device)
        return self._device_bases[key]

    def _make_hook(self, layer: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            if not self.enabled:
                return None
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(hidden):
                raise TypeError(f"layer {layer} output has unexpected type {type(output).__name__}")
            self.calls[layer] += 1
            projected = project_out(hidden, self._basis_on(layer, hidden.device))
            if isinstance(output, tuple):
                return (projected, *output[1:])
            if isinstance(output, list):
                return [projected, *output[1:]]
            return projected

        return hook

    def attach(self, model: Any, module_path: str | None = None) -> str:
        """Register the hooks on ``model``'s blocks; returns the module path that was used."""
        if self._handles:
            raise RuntimeError("SubspaceAblation is already attached")
        _, path = get_decoder_layers(model, module_path)
        for layer in self.layers:
            block = resolve_qwen_layer(model, layer, path)
            self._handles.append(block.register_forward_hook(self._make_hook(layer)))
        return path

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextlib.contextmanager
    def suspended(self) -> Iterator[None]:
        """Temporarily switch the intervention off (e.g. to probe the un-ablated model)."""
        previous, self.enabled = self.enabled, False
        try:
            yield
        finally:
            self.enabled = previous

    def reset_calls(self) -> None:
        for layer in self.calls:
            self.calls[layer] = 0


@torch.no_grad()
def verify_ablation(
    model: Any,
    ablation: SubspaceAblation,
    input_ids: Any,
    attention_mask: Any | None = None,
    tolerance: float = 1e-2,
) -> dict[int, dict[str, float]]:
    """Check on real activations that the hooks sit where the SAE reads and remove the subspace.

    Runs the model twice (ablation off, then on) with a capture hook registered *after* the
    ablation hooks, so it sees exactly what the next block receives. Reports, per layer,
    ``‖h Q‖ / ‖h‖`` before and after. After must be ~0; before shows how much of the residual stream
    the subspace carried on this input. Raises if the subspace is not removed or a hook never fired.
    """
    _, path = get_decoder_layers(model)
    captured: dict[tuple[bool, int], Any] = {}
    handles = []
    state = {"enabled": ablation.enabled}

    def make_capture(layer: int):
        def capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured[(state["enabled"], layer)] = hidden.detach().to(torch.float32)

        return capture

    was_training = model.training
    model.eval()
    try:
        for layer in ablation.layers:
            handles.append(resolve_qwen_layer(model, layer, path).register_forward_hook(make_capture(layer)))
        ablation.reset_calls()
        previous = ablation.enabled
        for enabled in (False, True):
            ablation.enabled = enabled
            state["enabled"] = enabled
            kwargs = {"input_ids": input_ids, "use_cache": False}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            model(**kwargs)
        ablation.enabled = previous
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    report: dict[int, dict[str, float]] = {}
    for layer in ablation.layers:
        if ablation.calls[layer] == 0:
            raise RuntimeError(f"CAFT hook on layer {layer} never fired during the verification forward pass")
        q = ablation.bases[layer].to(captured[(True, layer)].device)
        ratios = {}
        for enabled in (False, True):
            h = captured[(enabled, layer)]
            ratios[enabled] = ((h @ q).norm() / h.norm().clamp_min(1e-12)).item()
        report[layer] = {"fraction_in_subspace_before": ratios[False], "fraction_in_subspace_after": ratios[True]}
        if ratios[True] > tolerance:
            raise RuntimeError(
                f"layer {layer}: {ratios[True]:.3e} of the residual norm is still inside the ablated "
                f"subspace after the hook (tolerance {tolerance:.0e}); the hook is not on the SAE's activation"
            )
    return report
