"""Configuration contract tests."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from sae_caft.utils import (
    discover_pretrained_saes,
    load_config,
    publish_results,
    resolve_hf_repo_id,
    sample_rows,
)


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

    def test_discover_pretrained_saes_returns_available_layer_k_pairs(self) -> None:
        entries = {
            "resid_post_layer_2/trainer_1/config.json": {
                "trainer": {"layer": 2, "k": 64, "submodule_name": "resid_post_layer_2"}
            },
            "resid_post_layer_8/trainer_3/config.json": {
                "trainer": {"layer": 8, "k": 128, "submodule_name": "resid_post_layer_8"}
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            local_paths = {}
            for filename, contents in entries.items():
                local_path = Path(temp_dir) / filename
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_text(json.dumps(contents), encoding="utf-8")
                local_paths[filename] = str(local_path)

            fake_hub = types.ModuleType("huggingface_hub")

            class FakeHfApi:
                def list_repo_files(self, repo_id: str) -> list[str]:
                    self.repo_id = repo_id
                    return list(entries)

            fake_hub.HfApi = FakeHfApi
            fake_hub.hf_hub_download = lambda repo_id, filename: local_paths[filename]
            with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
                found = discover_pretrained_saes({"repo_id": "test/saes"})

        self.assertEqual(found, [(2, 64, 1), (8, 128, 3)])

    def test_resolve_hf_repo_id_accepts_username_repo_and_url(self) -> None:
        self.assertEqual(resolve_hf_repo_id({"username": "researcher"}), "researcher/sae-method-1")
        self.assertEqual(resolve_hf_repo_id({"repo": "researcher/results"}), "researcher/results")
        self.assertEqual(
            resolve_hf_repo_id({"repo": "https://huggingface.co/researcher/results"}),
            "researcher/results",
        )

    def test_publish_results_creates_repo_and_uploads_folder(self) -> None:
        calls = []
        fake_hub = types.ModuleType("huggingface_hub")

        class FakeHfApi:
            def create_repo(self, **kwargs) -> None:
                calls.append(("create_repo", kwargs))

            def upload_folder(self, **kwargs) -> None:
                calls.append(("upload_folder", kwargs))

        fake_hub.HfApi = FakeHfApi
        config = {
            "outputs": {
                "huggingface": {"enabled": True, "repo": "researcher/results", "private": True},
            }
        }
        with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
            url = publish_results(Path("/tmp/results"), config, "method_1/layer_15_k64")

        self.assertEqual(
            url,
            "https://huggingface.co/researcher/results/tree/main/method_1/layer_15_k64",
        )
        self.assertEqual([call[0] for call in calls], ["create_repo", "upload_folder"])
        self.assertEqual(calls[1][1]["path_in_repo"], "method_1/layer_15_k64")


if __name__ == "__main__":
    unittest.main()