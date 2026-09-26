"""Configuration contract tests."""

import unittest
from pathlib import Path

from sae_caft.utils import load_config


class ConfigTests(unittest.TestCase):
    def test_method_1_config_uses_base_model_and_training_split(self) -> None:
        config = load_config(Path(__file__).parents[1] / "config.yaml")
        self.assertEqual(config["model"]["name"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(config["sae"]["layer"], 15)
        self.assertEqual(config["sae"]["k"], 64)
        self.assertEqual(config["dataset"]["split"], "train")
        self.assertIn("bad_medical_advice.jsonl", config["dataset"]["path"])
        self.assertEqual(config["runtime"]["batch_size"], 1)


if __name__ == "__main__":
    unittest.main()