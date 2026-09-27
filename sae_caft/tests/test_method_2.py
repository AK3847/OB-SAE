"""CPU tests for Method-2 prompt sampling and cached response filtering."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from sae_caft.generate_chat_dataset import load_cached_chat_examples, sample_lmsys_prompts
from sae_caft.utils import response_activation_mask


class Method2Tests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed in the selected interpreter")
    def test_response_mask_uses_activation_positions_that_predict_targets(self) -> None:
        labels = [-100, -100, 31, 32, 33]

        response_mask = response_activation_mask(labels, valid_tokens=3, device="cpu")

        self.assertEqual(response_mask.tolist(), [[False, True, True, True, False]])
        with self.assertRaises(RuntimeError):
            response_activation_mask(labels, valid_tokens=4, device="cpu")

    def test_lmsys_sampling_is_repeatable_and_uses_unique_rows(self) -> None:
        dataset = [
            {"conversation": [{"role": "user", "content": f"prompt {index}"}]}
            for index in range(12)
        ]

        first = sample_lmsys_prompts(dataset, 5, seed=17)
        second = sample_lmsys_prompts(dataset, 5, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len({row["prompt_id"] for row in first}), 5)

    def test_cache_filters_short_responses_and_checks_generation_count(self) -> None:
        records = [
            {"prompt_id": 1, "prompt": "one", "response": "x" * 100, "response_length_chars": 100},
            {"prompt_id": 2, "prompt": "two", "response": "short", "response_length_chars": 5},
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "generations.jsonl"
            cache_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            config = {
                "method_2": {
                    "cache_path": str(cache_path),
                    "sample_size": 2,
                    "min_response_chars": 100,
                    "paper_reference_valid_examples": 1637,
                }
            }

            usable, resolved_path = load_cached_chat_examples(config)

        self.assertEqual(resolved_path, cache_path)
        self.assertEqual([row["prompt_id"] for row in usable], [1])


if __name__ == "__main__":
    unittest.main()