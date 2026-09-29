"""CPU tests for CAFT Method-3 activation-difference SAE semantics."""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from sae_caft.get_activation_difference import encode_masked_difference, resolve_method3_layer
    from sae_caft.get_attribution_chat import parse_int_list
    from sae_caft.utils import FrozenBatchTopKSAE


@unittest.skipIf(torch is None, "torch is not installed in the selected interpreter")
class Method3Tests(unittest.TestCase):
    def make_sae(self):
        return FrozenBatchTopKSAE(
            {
                "encoder.weight": torch.eye(2),
                "decoder.weight": torch.eye(2),
                "encoder.bias": torch.zeros(2),
                "bias": torch.zeros(2),
                "k": torch.tensor(2),
            },
            expected_k=2,
        )

    def test_difference_is_encoded_before_masked_aggregation(self):
        sae = self.make_sae()
        base = torch.tensor([[[2.0, 0.0], [5.0, 0.0]]])
        bad = torch.tensor([[[0.0, 3.0], [5.0, 0.0]]])
        mask = torch.tensor([[True, False]])

        latent_sum, token_count = encode_masked_difference(bad - base, mask, sae, token_chunk_size=1)

        self.assertEqual(token_count, 1)
        self.assertTrue(torch.equal(latent_sum, torch.tensor([0.0, 3.0], dtype=torch.float64)))
        wrong_method_4 = sae.encode(bad) - sae.encode(base)
        self.assertFalse(torch.equal(latent_sum, wrong_method_4[0, 0].to(dtype=torch.float64)))

    def test_list_and_scalar_layer_cli_values_parse(self):
        self.assertEqual(parse_int_list("[13,17]"), [13, 17])
        self.assertEqual(parse_int_list("15"), [15])
        self.assertEqual(parse_int_list("[64, 128]"), [64, 128])

    def test_layer_resolution_descends_through_peft_base_model(self):
        class BaseModel:
            def __init__(self):
                self.model = type("Transformer", (), {"layers": ["block_0", "block_1"]})()

        class PeftWrapper:
            def get_base_model(self):
                return BaseModel()

        self.assertEqual(resolve_method3_layer(PeftWrapper(), 1), "block_1")

if __name__ == "__main__":
    unittest.main()