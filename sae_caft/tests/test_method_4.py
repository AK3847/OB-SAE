import unittest

import torch

from sae_caft.get_latent_activation_difference import compute_method4, encode_masked_latent_sum
from sae_caft.utils import FrozenBatchTopKSAE


class Method4Tests(unittest.TestCase):
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

    def test_method4_latent_difference_matches_mean_encoder_difference(self):
        sae = self.make_sae()
        base = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]], dtype=torch.float32)
        bad = torch.tensor([[[0.0, 3.0], [1.0, 0.0]]], dtype=torch.float32)
        mask = torch.tensor([[True, False]], dtype=torch.bool)

        base_sum, bad_sum, token_count = encode_masked_latent_sum(
            base, bad, mask, sae, token_chunk_size=1
        )

        self.assertEqual(token_count, 1)
        self.assertTrue(torch.equal(base_sum, torch.tensor([1.0, 0.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(bad_sum, torch.tensor([0.0, 3.0], dtype=torch.float64)))

        delta = bad_sum - base_sum
        self.assertTrue(torch.equal(delta, torch.tensor([-1.0, 3.0], dtype=torch.float64)))
        self.assertTrue(torch.all(delta > 0) == torch.tensor(False))


if __name__ == "__main__":
    unittest.main()
