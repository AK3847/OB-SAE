"""CPU tests for released BatchTopK weight layout and attribution dimensions."""

import unittest

try:
    import torch
except ImportError:
    torch = None

from sae_caft.utils import FrozenBatchTopKSAE, calculate_attribution


@unittest.skipIf(torch is None, "torch is not installed in the selected interpreter")
class SAEShapeTests(unittest.TestCase):
    def test_encode_and_decoder_projection_shapes(self) -> None:
        state = {
            "encoder.weight": torch.randn(8, 4),
            "decoder.weight": torch.randn(4, 8),
            "encoder.bias": torch.zeros(8),
            "b_dec": torch.zeros(4),
            "k": torch.tensor(2),
            "threshold": torch.tensor(-1.0),
        }
        sae = FrozenBatchTopKSAE(state, expected_k=2)
        activation = torch.randn(1, 3, 4)
        encoded = sae.encode(activation)
        self.assertEqual(tuple(encoded.shape), (1, 3, 8))
        self.assertTrue(torch.all((encoded != 0).sum(dim=-1) <= 2))
        score = calculate_attribution(
            activation,
            torch.randn_like(activation),
            sae,
            torch.zeros((1, 3), dtype=torch.bool),
            chunk_size=3,
        )
        self.assertEqual(tuple(score.shape), (8,))


if __name__ == "__main__":
    unittest.main()