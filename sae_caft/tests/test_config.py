"""Configuration contract tests."""

import unittest
from pathlib import Path

from sae_caft.utils import load_config, sample_rows


class ConfigTests(unittest.TestCase):
    def test_method_1_config_uses_base_model_and_training_split(self) -> None:
        config = load_config(Path(__file__).parents[1] / "config.yaml")
        self.assertEqual(config["model"]["name"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertTrue(config["model"]["quantization"]["enabled"])
        self.assertEqual(config["model"]["quantization"]["type"], "4bit")
        self.assertEqual(config["sae"]["layer"], 15)
        self.assertEqual(config["sae"]["k"], 64)
        self.assertEqual(
            config["sae"]["trainer_directory_pattern"].format(layer=15, index=1),
            "resid_post_layer_15/trainer_1",
        )
        self.assertEqual(config["dataset"]["split"], "train")
        self.assertEqual(config["dataset"]["sft_eval_fraction"], 0.1)
        self.assertEqual(config["dataset"]["sample_seed"], 0)
        self.assertIn("bad_medical_advice.jsonl", config["dataset"]["path"])
        self.assertEqual(config["runtime"]["batch_size"], 1)

    def test_sample_rows_is_repeatable_and_returns_exact_count(self) -> None:
        rows = [{"id": index} for index in range(20)]
        first = sample_rows(rows, 7, seed=123)
        second = sample_rows(rows, 7, seed=123)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertEqual(len({row["id"] for row in first}), 7)

    def test_sample_rows_rejects_count_larger_than_available(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 2 are available"):
            sample_rows([{"id": 1}, {"id": 2}], 3, seed=0)


if __name__ == "__main__":
    unittest.main()