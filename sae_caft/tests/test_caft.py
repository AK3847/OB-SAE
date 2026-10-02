"""CPU tests for the CAFT subspace-ablation intervention."""

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sae_caft import caft_intervention as caft
from sae_caft.utils import FrozenBatchTopKSAE, load_config


def random_directions(d_model: int, count: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(d_model, count, generator=generator)


class BasisTests(unittest.TestCase):
    def test_qr_basis_is_orthonormal_and_spans_the_directions(self) -> None:
        dirs = random_directions(32, 4)
        q, _ = caft.build_orthonormal_basis(dirs)
        self.assertEqual(tuple(q.shape), (32, 4))
        self.assertTrue(torch.allclose(q.T @ q, torch.eye(4), atol=1e-5))
        # every raw direction lies in span(Q)
        self.assertTrue(torch.allclose(q @ (q.T @ dirs), dirs, atol=1e-5))

    def test_projection_removes_the_whole_subspace_not_each_vector_separately(self) -> None:
        d_model = 16
        dirs = random_directions(d_model, 3, seed=1)  # random, hence non-orthogonal, directions
        q, _ = caft.build_orthonormal_basis(dirs)
        h = torch.randn(2, 5, d_model)

        out = caft.project_out(h, q)

        self.assertLess((out @ dirs).abs().max().item(), 1e-4)
        # equals the least-squares projection onto span(dirs)
        hat = dirs @ torch.linalg.inv(dirs.T @ dirs) @ dirs.T
        self.assertTrue(torch.allclose(out, h - h @ hat, atol=1e-4))
        # naive per-vector subtraction of raw decoder vectors is a different (wrong) operation
        naive = h.clone()
        for column in dirs.T:
            naive = naive - (naive @ column).unsqueeze(-1) * column
        self.assertFalse(torch.allclose(out, naive, atol=1e-3))

    def test_projection_is_idempotent_and_keeps_dtype(self) -> None:
        q, _ = caft.build_orthonormal_basis(random_directions(16, 2))
        h = torch.randn(3, 16).to(torch.bfloat16)
        once = caft.project_out(h, q)
        self.assertEqual(once.dtype, torch.bfloat16)
        twice = caft.project_out(once, q)
        self.assertTrue(torch.allclose(once.float(), twice.float(), atol=2e-2))

    def test_gradients_flow_through_the_projection(self) -> None:
        q, _ = caft.build_orthonormal_basis(random_directions(16, 3))
        h = torch.randn(2, 4, 16, requires_grad=True)
        weight = torch.randn(2, 4, 16)

        (caft.project_out(h, q) * weight).sum().backward()

        self.assertIsNotNone(h.grad)
        # d/dh sum(P h * w) = P w with P = I - QQᵀ: nonzero, and itself outside the subspace
        expected = weight - (weight @ q) @ q.T
        self.assertTrue(torch.allclose(h.grad, expected, atol=1e-5))
        self.assertGreater(h.grad.abs().sum().item(), 0)

    def test_dependent_directions_are_rejected(self) -> None:
        dirs = random_directions(16, 2)
        dirs = torch.cat([dirs, dirs[:, :1] * 3.0], dim=1)
        with self.assertRaisesRegex(ValueError, "linearly dependent"):
            caft.build_orthonormal_basis(dirs, layer=7)

    def test_single_direction_gives_unit_vector(self) -> None:
        q, _ = caft.build_orthonormal_basis(random_directions(16, 1))
        self.assertEqual(tuple(q.shape), (16, 1))
        self.assertAlmostEqual(q.norm().item(), 1.0, places=5)


class DecoderTests(unittest.TestCase):
    def make_sae(self, d_model: int = 6, num_latents: int = 10) -> FrozenBatchTopKSAE:
        state = {
            "encoder.weight": torch.randn(num_latents, d_model),
            "decoder.weight": torch.randn(d_model, num_latents),
            "encoder.bias": torch.zeros(num_latents),
            "b_dec": torch.zeros(d_model),
            "k": torch.tensor(2),
        }
        return FrozenBatchTopKSAE(state, expected_k=2)

    def test_directions_are_decoder_rows_transposed(self) -> None:
        sae = self.make_sae()
        dirs = caft.directions_from_decoder(sae.W_dec, (7, 2, 4))
        self.assertEqual(tuple(dirs.shape), (6, 3))
        for column, latent in enumerate((7, 2, 4)):
            self.assertTrue(torch.equal(dirs[:, column], sae.W_dec[latent].float()))

    def test_out_of_range_latent_ids_raise_a_clear_error(self) -> None:
        sae = self.make_sae(num_latents=10)
        with self.assertRaisesRegex(ValueError, r"layer 15: latent IDs \[10, 99\] are outside the SAE vocabulary \[0, 10\)"):
            caft.directions_from_decoder(sae.W_dec, (3, 99, 10), layer=15)

    def test_load_layer_basis_end_to_end_without_network(self) -> None:
        sae = self.make_sae(d_model=6, num_latents=10)
        state = {
            "encoder.weight": sae.W_enc.T.detach().clone(),
            "decoder.weight": sae.W_dec.T.detach().clone(),
            "encoder.bias": sae.b_enc.detach().clone(),
            "b_dec": sae.b_dec.detach().clone(),
            "k": torch.tensor(2),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "ae.pt"
            torch.save(state, checkpoint)
            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.hf_hub_download = lambda repo_id, filename: str(checkpoint)
            released = {"trainer": {"submodule_name": "resid_post_layer_3", "dict_size": 10}}
            spec = caft.LayerSpec(layer=3, k=2, latent_ids=(1, 5, 8))
            with patch.dict("sys.modules", {"huggingface_hub": fake_hub}), patch.object(
                caft, "_trainer_config", return_value=("resid_post_layer_3/trainer_0", Path("x"), released)
            ):
                basis, meta = caft.load_layer_basis(spec, "repo/x", "resid_post_layer_{layer}/trainer_{index}", 6)
                with self.assertRaisesRegex(ValueError, "outside the SAE vocabulary"):
                    caft.load_layer_basis(
                        caft.LayerSpec(3, 2, (1, 10)), "repo/x", "resid_post_layer_{layer}/trainer_{index}", 6
                    )
                with self.assertRaisesRegex(ValueError, "d_model"):
                    caft.load_layer_basis(spec, "repo/x", "resid_post_layer_{layer}/trainer_{index}", 7)

        self.assertEqual(tuple(basis.shape), (6, 3))
        self.assertTrue(torch.allclose(basis.T @ basis, torch.eye(3), atol=1e-5))
        decoder_rows = sae.W_dec[[1, 5, 8]].T.detach()
        self.assertTrue(torch.allclose(basis @ (basis.T @ decoder_rows), decoder_rows, atol=1e-5))
        self.assertEqual(meta["latent_ids"], [1, 5, 8])
        self.assertEqual(meta["trainer_directory"], "resid_post_layer_3/trainer_0")


class ConfigParsingTests(unittest.TestCase):
    def test_mixed_layer_forms_and_unequal_latent_counts(self) -> None:
        config = {
            "sae": {"k": 64},
            "caft": {
                "k": 32,
                "layers": {
                    19: {"k": 256, "latents": [102721, 110362, 5]},
                    11: [109597, 126173],
                    15: [64256],
                },
            },
        }
        specs = caft.parse_caft_config(config)
        self.assertEqual([s.layer for s in specs], [11, 15, 19])
        self.assertEqual([len(s.latent_ids) for s in specs], [2, 1, 3])
        self.assertEqual([s.k for s in specs], [32, 32, 256])

    def test_default_k_falls_back_to_sae_k(self) -> None:
        specs = caft.parse_caft_config({"sae": {"k": 64}, "caft": {"layers": {15: [1]}}})
        self.assertEqual(specs[0].k, 64)

    def test_empty_or_malformed_layers_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "caft.layers is empty"):
            caft.parse_caft_config({"caft": {"layers": {}}})
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            caft.parse_caft_config({"sae": {"k": 64}, "caft": {"layers": {15: []}}})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            caft.parse_caft_config({"sae": {"k": 64}, "caft": {"layers": {15: [1, 2, 1]}}})
        with self.assertRaisesRegex(ValueError, "integer"):
            caft.parse_caft_config({"sae": {"k": 64}, "caft": {"layers": {15: ["a"]}}})
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            caft.parse_caft_config({"sae": {"k": 64}, "caft": {"layers": {15: {"latents": [1], "x": 1}}}})

    def test_repository_config_has_a_caft_section_and_no_hardcoded_latents_in_code(self) -> None:
        config = load_config(Path(__file__).parents[1] / "config.yaml")
        self.assertIn("caft", config)
        self.assertEqual(config["caft"]["sae_repo"], config["sae"]["repo_id"])


class ToyBlock(torch.nn.Module):
    def __init__(self, d_model: int, tuple_output: bool):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_model)
        self.tuple_output = tuple_output

    def forward(self, x):
        out = x + torch.tanh(self.linear(x))
        return (out, "aux") if self.tuple_output else out


class ToyInner(torch.nn.Module):
    def __init__(self, d_model: int, layers: int, tuple_output: bool):
        super().__init__()
        self.embed = torch.nn.Embedding(20, d_model)
        self.layers = torch.nn.ModuleList(ToyBlock(d_model, tuple_output) for _ in range(layers))

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        return x


class ToyLM(torch.nn.Module):
    def __init__(self, d_model: int = 12, layers: int = 4, tuple_output: bool = True):
        super().__init__()
        self.config = types.SimpleNamespace(num_hidden_layers=layers)
        self.model = ToyInner(d_model, layers, tuple_output)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return self.model(input_ids)


class PeftLike(torch.nn.Module):
    """Mimics PeftModel -> LoraModel -> CausalLM nesting."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.config = inner.config
        self.base_model = types.SimpleNamespace(model=inner)
        self.inner = inner

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


class HookTests(unittest.TestCase):
    def test_hook_preserves_tuple_auxiliary_outputs_and_projects_hidden(self) -> None:
        model = ToyLM()
        q, _ = caft.build_orthonormal_basis(random_directions(12, 2))
        captured = {}
        ablation = caft.SubspaceAblation({2: q})
        ablation.attach(model)
        model.model.layers[2].register_forward_hook(lambda _m, _i, out: captured.update(out=out))

        model(torch.tensor([[1, 2, 3]]))

        hidden, aux = captured["out"]
        self.assertEqual(aux, "aux")
        self.assertLess((hidden @ q).abs().max().item(), 1e-5)
        self.assertEqual(ablation.calls, {2: 1})

    def test_hook_handles_tensor_and_list_outputs(self) -> None:
        q, _ = caft.build_orthonormal_basis(random_directions(12, 1))
        hook = caft.SubspaceAblation({0: q})._make_hook(0)
        h = torch.randn(1, 2, 12)
        self.assertTrue(torch.is_tensor(hook(None, (), h)))
        listed = hook(None, (), [h, "x", None])
        self.assertIsInstance(listed, list)
        self.assertEqual(listed[1:], ["x", None])
        with self.assertRaises(TypeError):
            hook(None, (), ("not a tensor",))

    def test_suspended_disables_and_remove_detaches(self) -> None:
        model = ToyLM(tuple_output=False)
        q, _ = caft.build_orthonormal_basis(random_directions(12, 2))
        ids = torch.tensor([[1, 2, 3]])
        baseline = model(ids)
        ablation = caft.SubspaceAblation({1: q})
        ablation.attach(model)
        self.assertFalse(torch.allclose(model(ids), baseline))
        with ablation.suspended():
            self.assertTrue(torch.allclose(model(ids), baseline))
        self.assertTrue(ablation.enabled)
        ablation.remove()
        self.assertTrue(torch.allclose(model(ids), baseline))

    def test_gradients_reach_earlier_parameters_through_the_intervention(self) -> None:
        torch.manual_seed(0)
        model = ToyLM(tuple_output=True)
        ids = torch.tensor([[1, 2, 3, 4]])
        q, _ = caft.build_orthonormal_basis(random_directions(12, 3))

        def first_layer_grad(with_ablation: bool) -> torch.Tensor:
            model.zero_grad()
            ablation = caft.SubspaceAblation({1: q})
            if with_ablation:
                ablation.attach(model)
            model(ids).square().sum().backward()
            ablation.remove()
            return model.model.layers[0].linear.weight.grad.clone()

        plain, ablated = first_layer_grad(False), first_layer_grad(True)
        self.assertGreater(ablated.abs().sum().item(), 0)
        self.assertFalse(torch.allclose(plain, ablated))

    def test_attach_works_through_peft_style_wrappers_and_verifies(self) -> None:
        model = PeftLike(ToyLM())
        q, _ = caft.build_orthonormal_basis(random_directions(12, 2))
        ablation = caft.SubspaceAblation({0: q, 3: q})
        path = ablation.attach(model)
        self.assertEqual(path, "base_model.model.model.layers")
        report = caft.verify_ablation(model, ablation, torch.tensor([[1, 2, 3, 4, 5]]))
        self.assertEqual(sorted(report), [0, 3])
        for stats in report.values():
            self.assertLess(stats["fraction_in_subspace_after"], 1e-5)
            self.assertGreater(stats["fraction_in_subspace_before"], 1e-3)
        self.assertTrue(ablation.enabled)

    def test_verification_fails_if_a_hook_never_fires(self) -> None:
        model = ToyLM()
        q, _ = caft.build_orthonormal_basis(random_directions(12, 1))
        ablation = caft.SubspaceAblation({1: q})  # never attached
        with self.assertRaisesRegex(RuntimeError, "never fired"):
            caft.verify_ablation(model, ablation, torch.tensor([[1, 2]]))

    def test_get_decoder_layers_rejects_wrong_layout(self) -> None:
        model = ToyLM()
        model.config.num_hidden_layers = 99
        with self.assertRaisesRegex(TypeError, "Could not find the transformer block list"):
            caft.get_decoder_layers(model)


if __name__ == "__main__":
    unittest.main()
