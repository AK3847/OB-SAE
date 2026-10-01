"""Offline regression coverage for Method-1 interpretation and its prompts."""

import csv
import json
import math
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

from sae_caft import interpret_method1 as method
from sae_caft import interpretation_prompts as prompts


@dataclass
class FakeTokenLogprob:
    token: str
    logprob: float


def logprob(token: str, probability: float) -> FakeTokenLogprob:
    return FakeTokenLogprob(token=token, logprob=math.log(probability))


def required_json(path: Path) -> Any:
    result = method.read_json(path)
    assert result is not None, f"Expected valid JSON at {path}"
    return result


def response(text, probabilities=None):
    content = None if probabilities is None else [
        SimpleNamespace(top_logprobs=[logprob(token, probability)
                                     for token, probability in probabilities])
    ]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                 logprobs=SimpleNamespace(content=content))],
        model_dump=lambda: {"text": text, "probabilities": probabilities},
    )


def api_client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class APIError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class FakeTokenizer:
    eos_token_id = None

    def __call__(self, text, **kwargs):
        return {"input_ids": [int(word) for word in text.split()][:kwargs["max_length"]]}

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return "".join(f" token{token}" for token in ids)


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Identity()
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask, use_cache):
        self.forward_calls += 1
        hidden = torch.stack((input_ids.float(), torch.zeros_like(input_ids).float()), dim=-1)
        return self.layer(hidden)


class FakeModel:
    def __init__(self):
        self.model = FakeBackbone()
        self.embeddings = torch.nn.Embedding(16, 2)

    def get_input_embeddings(self):
        return self.embeddings


class FakeSAE:
    def __init__(self, k):
        self.W_enc = torch.eye(2)
        self.k = k
        self.calls = []

    def encode(self, hidden):
        self.calls.append(hidden.clone())
        dense = hidden @ self.W_enc
        values, indices = dense.topk(self.k, dim=-1)
        return torch.zeros_like(dense).scatter(-1, indices, values)


def config_for(output):
    return {
        "model": {"name": "offline-qwen"},
        "sae": {"activation_dim": 2, "latent_dim": 2, "module_path": "model.layers",
                "layer": 0, "k": 1},
        "dataset": {"max_seq_length": 4},
        "runtime": {"seed": 17},
        "outputs": {"huggingface": {"repo_type": "model"}},
        "method_1_interpretation": {
            "layers": [0], "k_values": [1], "top_n": 1, "examples_per_latent": 5,
            "context_length": 2, "encoding_token_chunk_size": 3,
            "hf_repo": "offline/repo", "hf_subdir": "/method1/",
            "fineweb_dataset": "offline/fineweb", "fineweb_config": "sample",
            "fineweb_split": "train", "fineweb_revision": "offline-revision",
            "fineweb_samples": 1, "max_tokens": 10,
            "explainer_model": "offline-judge", "explanation_max_tokens": 128,
            "use_logprobs": True, "output_directory": str(output),
        },
    }


class PromptTests(unittest.TestCase):
    def test_highlight_contiguous_sign_zero_and_ties(self):
        cases = [
            (list("abcdef"), [0, 2, 1, 0, 3, -1], "a<<bc>>d<<e>>f"),
            (list("abcdef"), [0, -2, -1, 0, -3, 1], "a<<bc>>d<<e>>f"),
            (["a", " b", " c"], [2, -2, 1], "<<a>> b<< c>>"),
            (["a", " b", " c"], [-2, 2, -1], "<<a>> b<< c>>"),
            (["a", " b"], [0, 0], "a b"),
            ([" x", "y", "!"], [1, 2, 3], "<< xy!>>"),
            ([], [], ""),
        ]
        for tokens, activations, expected in cases:
            with self.subTest(activations=activations):
                self.assertEqual(prompts.highlight(tokens, activations), expected)
        with self.assertRaisesRegex(ValueError, "same length"):
            prompts.highlight(["x"], [])

    def test_exact_three_fewshot_pairs_and_numbered_input(self):
        # Independent literals protect against changes to the demonstrations themselves.
        expected = [
            ("Example 1: and he was <<over the moon>> to find\n"
             "Example 2: we'll be laughing <<till the cows come home>>! Pro\n"
             "Example 3: thought Scotland was boring, but really there's more <<than meets the eye>>! I'd",
             r"\boxed{Common idioms in text conveying positive sentiment.}"),
            ("Example 1: a river is wide but the ocean is wid<<er>>. The ocean\n"
             'Example 2: every year you get tall<<er>>," she\n'
             "Example 3: the hole was small<<er>> but deep<<er>> than the",
             r'\boxed{The token "er" at the end of a comparative adjective describing size.}'),
            ('Example 1: something happening inside my <<house>>", he\n'
             'Example 2: presumably was always contained in <<a box>>", according\n'
             'Example 3: people were coming into the <<smoking area>>".\n\n'
             'However he\nExample 4: Patrick: "why are you getting in the << way?>>" Later,',
             r"\boxed{Nouns representing distinct objects that contain something, sometimes preceding a quotation mark.}"),
        ]
        self.assertEqual(prompts.EXAMPLES, expected)
        examples = [{"highlighted_text": "<<one>>", "mean_attribution": -9876.5},
                    {"highlighted_text": "two <<three>>", "latent_id": 1234}]
        messages = prompts.explanation_messages(examples)
        self.assertEqual(len(messages), 8)
        self.assertEqual(messages[0], {"role": "system", "content": prompts.SYSTEM})
        for index, (text, explanation) in enumerate(expected):
            self.assertEqual(messages[1 + 2 * index], {"role": "user", "content": text})
            self.assertEqual(messages[2 + 2 * index], {"role": "assistant", "content": explanation})
        self.assertEqual(messages[-1], {"role": "user", "content":
                                       "Example 1: <<one>>\nExample 2: two <<three>>"})
        self.assertNotIn("9876", json.dumps(messages))
        self.assertNotIn("1234", json.dumps(messages))

    def test_boxed_parse_matches_first_non_multiline_box(self):
        cases = [(r"reasoning \boxed{  useful pattern  } tail", "useful pattern"),
                 (r"\boxed{first} \boxed{second}", "first"),
                 (r"\boxed{}", ""), ("unboxed", None),
                 ("\\boxed{line\nbreak}", None),
                 (r"\boxed{outer {inner} rest}", "outer {inner")]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(prompts.parse_explanation(raw), expected)

    def test_probability_weighted_score_and_numeric_mass_cutoff(self):
        score = prompts.weighted_score([
            logprob("0", .1), logprob("100", .3), logprob("refusal", .6)])
        assert score is not None
        self.assertAlmostEqual(score, 75.0)
        self.assertEqual(prompts.weighted_score([logprob("80", .25)]), 80.0)
        self.assertIsNone(prompts.weighted_score([logprob("80", .249)]))
        self.assertIsNone(prompts.weighted_score([]))
        self.assertIsNone(prompts.weighted_score([
            logprob("101", .3), logprob("-1", .3), logprob("50.0", .3)]))
        score = prompts.weighted_score([
            logprob(" 20", .1), logprob("+020", .2), logprob("100", .1)])
        assert score is not None
        self.assertAlmostEqual(score, 40.0)
        # Exact duplicate token strings overwrite, rather than adding probability.
        self.assertIsNone(prompts.weighted_score([logprob("50", .4), logprob("50", .1)]))

    def test_relevance_prompt_preserves_task_and_medical_safety_distinction(self):
        text = "Example 1: TASK_SENTINEL"
        rendered = prompts.relevance_prompt(text)
        self.assertIn("[TASK EXAMPLES START]\n" + text + "\n[TASK EXAMPLES END]", rendered)
        self.assertIn(prompts.RELEVANCE_POINTERS, rendered)
        self.assertIn("Medical vocabulary or medical subject matter alone is not evidence", rendered)
        self.assertIn("single number between 0 and 100", rendered)


class ScoringTests(unittest.TestCase):
    def test_weighted_query_options_and_first_generated_token_only(self):
        answer = response("100", [("0", .1), ("100", .3)])
        answer.choices[0].logprobs.content.append(
            SimpleNamespace(top_logprobs=[logprob("0", 1.0)]))
        create = Mock(return_value=answer)
        messages = [{"role": "user", "content": "score"}]
        score, raw, mode, payload = method.query_relevance(api_client(create), "judge", messages, True, 7)
        assert score is not None
        self.assertAlmostEqual(score, 75)
        self.assertEqual((raw, mode, payload), ("100", "logprob_weighted", answer.model_dump()))
        create.assert_called_once_with(model="judge", messages=messages, temperature=0, seed=7,
                                       max_tokens=1, logprobs=True, top_logprobs=20)

    def test_missing_or_low_mass_logprobs_do_not_trigger_sampled_fallback(self):
        for answer in (response("80"), response("80", [("80", .1), ("no", .9)])):
            with self.subTest(answer=answer):
                create = Mock(return_value=answer)
                score, _, mode, _ = method.query_relevance(api_client(create), "judge", [], True, 0)
                self.assertIsNone(score)
                self.assertEqual(mode, "logprob_weighted")
                self.assertEqual(create.call_count, 1)

    def test_explicit_unsupported_logprob_error_allows_one_fallback(self):
        for status, message in [(400, "logprobs not supported"),
                                (422, "unsupported parameter: top_logprobs")]:
            with self.subTest(status=status):
                create = Mock(side_effect=[APIError(message, status), response(" 91\n")])
                result = method.query_relevance(api_client(create), "judge", [], True, 0)
                self.assertEqual(result[:3], (91.0, " 91\n", "sampled_integer_fallback"))
                self.assertEqual(create.call_count, 2)
                fallback = create.call_args.kwargs
                self.assertEqual(fallback["max_tokens"], 4)
                self.assertNotIn("logprobs", fallback)
                self.assertNotIn("top_logprobs", fallback)

    def test_unrelated_errors_are_propagated_without_retry(self):
        for error in [APIError("logprobs unsupported", 401),
                      APIError("logprobs unsupported", 429),
                      APIError("unsupported model", 400),
                      APIError("invalid max_tokens", 422),
                      RuntimeError("logprobs unsupported")]:
            with self.subTest(error=str(error), status=getattr(error, "status_code", None)):
                create = Mock(side_effect=error)
                with self.assertRaises(type(error)) as raised:
                    method.query_relevance(api_client(create), "judge", [], True, 0)
                self.assertIs(raised.exception, error)
                self.assertEqual(create.call_count, 1)

    def test_strict_sampled_integer_fallback(self):
        for text, expected in [("0", 0.0), ("100", 100.0), (" 42\n", 42.0),
                               ("09", 9.0), ("101", None), ("-1", None),
                               ("+2", None), ("2.0", None), ("42 percent", None),
                               ("42\n43", None), ("", None)]:
            with self.subTest(text=text):
                create = Mock(return_value=response(text))
                self.assertEqual(method.query_relevance(api_client(create), "judge", [], False, 0)[:3],
                                 (expected, text, "sampled_integer_fallback"))
                self.assertEqual(create.call_count, 1)
                self.assertNotIn("logprobs", create.call_args.kwargs)


class WindowTests(unittest.TestCase):
    def test_topk_nonoverlapping_stable_ties_and_zero_exclusion_per_candidate(self):
        heaps = {7: [], 9: []}
        selected = torch.tensor([[1., 0.], [4., 0.], [4., 1.], [2., 2.],
                                 [0., 3.], [0., 0.], [8., 0.], [1., 0.]])
        method.update_top_windows(heaps, list(range(8)), selected, [7, 9], 2, 2, 0)
        method.update_top_windows(heaps, [8, 9, 10, 11],
                                  torch.tensor([[4., 0.], [0., 0.], [0., 0.], [0., 0.]]),
                                  [7, 9], 2, 2, 8)
        first = sorted(heaps[7], reverse=True)
        second = sorted(heaps[9], reverse=True)
        self.assertEqual([(item[0], -item[1], item[2]) for item in first],
                         [(8.0, 3, [6, 7]), (4.0, 0, [0, 1])])
        self.assertEqual([(item[0], -item[1], item[2]) for item in second],
                         [(3.0, 2, [4, 5]), (2.0, 1, [2, 3])])
        for heap in heaps.values():
            ids = [token for entry in heap for token in entry[2]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(max(entry[3]) > 0 for entry in heap))

    def test_find_examples_encodes_all_columns_in_chunks_and_ranks_across_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.save({"token_ids": torch.tensor([1, 2, 3, 4]),
                        "residuals": torch.tensor([[1., 9.], [2., 8.], [5., 0.], [4., 0.]])},
                       root / "a.pt")
            torch.save({"token_ids": torch.tensor([5, 6, 7, 8]),
                        "residuals": torch.tensor([[5., 0.], [1., 0.], [0., 9.], [0., 8.]])},
                       root / "b.pt")
            sae = FakeSAE(1)
            found = method.find_examples(sae, FakeTokenizer(), root, {"shards": ["a.pt", "b.pt"]},
                                         [0, 1], {"context_length": 2, "examples_per_latent": 5,
                                                  "encoding_token_chunk_size": 3})
            self.assertEqual([len(chunk) for chunk in sae.calls], [3, 1, 3, 1])
            self.assertEqual([row["window_id"] for row in found[0]], [1, 2])
            self.assertEqual([row["window_id"] for row in found[1]], [0, 3])
            self.assertEqual(found[0][0]["highlighted_text"], "<< token3 token4>>")
            self.assertEqual(found[0][0]["window_rank"], 1)
            self.assertEqual(found[0][0]["maximum_activation"], 5.0)
            # Candidate 0 loses to the nonzero competing feature in the first window.
            self.assertNotIn(0, [row["window_id"] for row in found[0]])


class FileAndResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_candidates(self, rows):
        path = self.root / "candidates.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["rank", "latent_id", "mean_attribution", "ignored"])
            writer.writerows([list(row) + ["metadata"] for row in rows])
        return path

    def test_candidate_csv_sorted_rank_topn_not_attribution(self):
        path = self.write_candidates([(3, 4, 100), (1, 2, -5), (2, 3, 1), (4, 99, "nan")])
        self.assertEqual(method.load_candidates(path, 2, 5), [
            {"method1_rank": 1, "latent_id": 2, "mean_attribution": -5.0},
            {"method1_rank": 2, "latent_id": 3, "mean_attribution": 1.0}])

    def test_candidate_csv_rejects_invalid_selected_rows(self):
        cases = [([(1, 0, 1)], 2, "only 1 candidates"),
                 ([(1, 0, 1), (2, 0, 2)], 2, "Duplicate"),
                 ([(0, 0, 1)], 1, "Invalid"), ([(1, -1, 1)], 1, "Invalid"),
                 ([(1, 2, 1)], 1, "Invalid"), ([(1, 0, "nan")], 1, "Invalid"),
                 ([(1, 0, "inf")], 1, "Invalid")]
        for rows, top_n, error in cases:
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, error):
                    method.load_candidates(self.write_candidates(rows), top_n, 2)

    def test_completed_result_requires_matching_key_candidate_and_valid_score(self):
        path = self.root / "result.json"
        candidate = {"latent_id": 0, "method1_rank": 1, "mean_attribution": -5.0}
        result = dict(candidate, run_key="key", status="complete", explanation="pattern", relevance_score=75)
        method.write_json(path, result)
        self.assertEqual(method.completed_result(path, "key", candidate), result)
        self.assertIsNone(method.completed_result(path, "changed", candidate))
        for change in [{"status": "failed"}, {"explanation": ""}, {"explanation": None},
                       {"relevance_score": True}, {"relevance_score": "75"},
                       {"relevance_score": None}, {"relevance_score": -1},
                       {"relevance_score": 101}, {"latent_id": 1},
                       {"method1_rank": 2}, {"mean_attribution": -4.0}]:
            with self.subTest(change=change):
                method.write_json(path, dict(result, **change))
                self.assertIsNone(method.completed_result(path, "key", candidate))
        path.write_text("truncated {", encoding="utf-8")
        self.assertIsNone(method.completed_result(path, "key", candidate))
        self.assertIsNone(method.completed_result(self.root / "missing.json", "key", candidate))

    def test_examples_resume_validation(self):
        path = self.root / "examples.json"
        settings = {"context_length": 2, "examples_per_latent": 1}
        row = {"highlighted_text": "<<ab>>", "token_ids": [1, 2], "tokens": ["a", "b"],
               "activations": [1, 2]}
        saved = {"run_key": "key", "examples": [row]}
        method.write_json(path, saved)
        self.assertEqual(method.valid_examples(path, "key", settings), [row])
        self.assertIsNone(method.valid_examples(path, "changed", settings))
        for invalid in [[], ["not a dict"], [dict(row, highlighted_text=7)],
                        [dict(row, token_ids=[1])], [dict(row, tokens=[])],
                        [dict(row, activations=[1])], [row, row]]:
            with self.subTest(invalid=invalid):
                method.write_json(path, {"run_key": "key", "examples": invalid})
                self.assertIsNone(method.valid_examples(path, "key", settings))
        path.write_text("not json", encoding="utf-8")
        self.assertIsNone(method.valid_examples(path, "key", settings))

    def test_null_cached_example_arrays_are_rejected_not_raised(self):
        # A structurally corrupt cache should be a cache miss, not abort run().
        path = self.root / "examples.json"
        for key in ("token_ids", "tokens", "activations"):
            with self.subTest(key=key):
                row = {"highlighted_text": "<<ab>>", "token_ids": [1, 2],
                       "tokens": ["a", "b"], "activations": [1, 2]}
                row[key] = None
                method.write_json(path, {"run_key": "key", "examples": [row]})
                self.assertIsNone(method.valid_examples(path, "key",
                                 {"context_length": 2, "examples_per_latent": 1}))

    def test_fingerprint_stable_key_order_changes_with_settings(self):
        self.assertEqual(method.fingerprint({"a": 1, "b": 2}), method.fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(method.fingerprint({"seed": 1}), method.fingerprint({"seed": 2}))


class OfflineRunTests(unittest.TestCase):
    def test_method2_interpretation_source_rank_outputs_and_resume(self):
        for candidate_file in ("top_25.csv", "ranked_attribution.csv"):
            with self.subTest(candidate_file=candidate_file), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = config_for(root / "method1")
                config["method_2_interpretation"] = {
                    "output_directory": str(root / "method2"), "candidate_file": candidate_file,
                }
                method.select_method(config, 2)
                output = Path(config["method_1_interpretation"]["output_directory"])
                with self.environment(root) as env:
                    if candidate_file == "ranked_attribution.csv":
                        (root / "top_25.csv").write_text(
                            "latent_id,attribution_effect,rank\n0,-9876.5,1\n", encoding="utf-8")
                    self.assertEqual(method.run(config, output), 0)
                    env.hub.hf_hub_download.assert_called_once_with(
                        repo_id="offline/repo", filename=f"method_2/layer_0_k1/{candidate_file}",
                        repo_type="model")
                    result = required_json(output / "layer_0_k1/latent_0/result.json")
                    self.assertEqual(result["source_method"], 2)
                    self.assertEqual(result["method2_rank"], 1)
                    self.assertNotIn("method1_rank", result)
                    self.assertEqual(result["mean_attribution"], -9876.5)
                    self.assertEqual(required_json(output / "metadata.json")["source_method"], 2)
                    with (output / "results.csv").open() as stream:
                        row = next(csv.DictReader(stream))
                    self.assertEqual((row["source_method"], row["ranking_rank"], row["method2_rank"]),
                                     ("2", "1", "1"))
                    self.assertEqual(row["method1_rank"], "")
                    self.assertNotIn("-9876.5", json.dumps(env.calls[-1]["messages"]))
                    self.assertEqual(len(env.calls), 2)
                    self.assertEqual(method.run(config, output), 0)
                    self.assertEqual(len(env.calls), 2)
                    self.assertFalse((root / "method1").exists())

    def test_unused_method2_config_preserves_method1_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            config = config_for(output)
            with self.environment(root) as env:
                self.assertEqual(method.run(config, output), 0)
                config["method_2_interpretation"] = {"output_directory": "outputs/method_2/interpretation"}
                self.assertEqual(method.run(config, output), 0)
                self.assertEqual(len(env.calls), 2)

    def test_switching_methods_in_same_directory_does_not_resume_other_method(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            config = config_for(output)
            with self.environment(root) as env:
                self.assertEqual(method.run(config, output), 0)
                first = required_json(output / "layer_0_k1/latent_0/result.json")
                method.select_method(config, 2)
                self.assertEqual(method.run(config, output), 0)
                second = required_json(output / "layer_0_k1/latent_0/result.json")
                self.assertNotEqual(first["run_key"], second["run_key"])
                self.assertEqual(second["source_method"], 2)
                # An identical explanation prompt can safely reuse its checkpoint;
                # the completed Method-1 result is not reused and relevance runs again.
                self.assertEqual(len(env.calls), 3)
                self.assertEqual(env.calls[-1]["max_tokens"], 1)
                self.assertEqual(env.model.model.forward_calls, 3)

    @contextmanager
    def environment(self, root, score_probabilities=None):
        source = root / "top_25.csv"
        source.write_text("rank,latent_id,mean_attribution\n1,0,-9876.5\n", encoding="utf-8")
        model = FakeModel()
        calls = []
        saes = []
        probabilities = score_probabilities if score_probabilities is not None else [("0", .1), ("100", .3)]
        explainer = Mock(return_value=response(r"Analysis. \boxed{A recurring token pattern.}"))

        def create(**kwargs):
            # Snapshot requests: run() appends to the original messages list later.
            calls.append(deepcopy(kwargs))
            if kwargs.get("max_tokens") == 1:
                return response("100", probabilities)
            return explainer(**kwargs)

        def load_sae(config):
            sae = FakeSAE(config["sae"]["k"])
            saes.append(sae)
            return sae, {"offline": True, "k": sae.k}

        hub = ModuleType("huggingface_hub")
        setattr(hub, "hf_hub_download", Mock(return_value=str(source)))
        datasets = ModuleType("datasets")
        setattr(datasets, "load_dataset", Mock(return_value=[{"text": "1 2 3 4 5 6 7 8 9 10"}]))
        openai = ModuleType("openai")
        setattr(openai, "OpenAI", Mock(return_value=api_client(create)))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"huggingface_hub": hub, "datasets": datasets,
                                                        "openai": openai}))
            load_model = stack.enter_context(patch.object(method, "load_model", return_value=(model, FakeTokenizer())))
            load_tokenizer = stack.enter_context(patch.object(method, "load_tokenizer", return_value=FakeTokenizer()))
            sae_loader = stack.enter_context(patch.object(method, "load_sae", side_effect=load_sae))
            stack.enter_context(patch.object(method, "resolve_qwen_layer", side_effect=lambda m, *_: m.model.layer))
            stack.enter_context(patch.object(method, "resolve_hf_repo_id", return_value="offline/repo"))
            stack.enter_context(patch.object(method, "task_examples", return_value="Example 1: OFFLINE_TASK"))
            stack.enter_context(patch.object(method, "cleanup_memory"))
            stack.enter_context(patch.object(method, "set_reproducibility_seed"))
            yield SimpleNamespace(model=model, calls=calls, saes=saes, load_model=load_model,
                                  load_tokenizer=load_tokenizer, explainer=explainer,
                                  load_sae=sae_loader, hub=hub, datasets=datasets, openai=openai)

    def test_single_latent_five_examples_two_api_calls_outputs_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            config = config_for(output)
            original = deepcopy(config)
            with self.environment(root) as env:
                self.assertEqual(method.run(config, output), 0)
                self.assertEqual(config, original)
                self.assertEqual(env.model.model.forward_calls, 3)
                self.assertEqual(len(env.model.model.layer._forward_hooks), 0)
                env.load_model.assert_called_once()
                env.load_sae.assert_called_once()
                env.hub.hf_hub_download.assert_called_once_with(
                    repo_id="offline/repo", filename="method1/layer_0_k1/top_25.csv", repo_type="model")
                env.datasets.load_dataset.assert_called_once_with(
                    "offline/fineweb", "sample", split="train", streaming=True, revision="offline-revision")
                env.openai.OpenAI.assert_called_once_with(max_retries=5, timeout=120.0)
                self.assertEqual(len(env.calls), 2)
                explainer, judge = env.calls
                self.assertEqual(len(explainer["messages"]), 8)
                self.assertEqual(judge["messages"][:-2], explainer["messages"])
                self.assertEqual(judge["messages"][-2],
                                 {"role": "assistant", "content": r"Analysis. \boxed{A recurring token pattern.}"})
                self.assertIn("OFFLINE_TASK", judge["messages"][-1]["content"])
                for request in env.calls:
                    self.assertNotIn("mean_attribution", json.dumps(request["messages"]))
                    self.assertNotIn("9876.5", json.dumps(request["messages"]))
                    self.assertEqual(request["model"], "offline-judge")
                    self.assertEqual(request["seed"], 17)
                    self.assertEqual(request["temperature"], 0)
                self.assertEqual(judge["top_logprobs"], 20)
                latent = output / "layer_0_k1" / "latent_0"
                result = required_json(latent / "result.json")
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["explanation"], "A recurring token pattern.")
                self.assertAlmostEqual(result["relevance_score"], 75.0)
                self.assertEqual(result["num_examples"], 5)
                self.assertEqual(result["mean_attribution"], -9876.5)
                self.assertEqual(result["scoring_mode"], "logprob_weighted")
                saved = required_json(latent / "examples.json")
                self.assertEqual(saved["run_key"], result["run_key"])
                self.assertEqual([row["window_id"] for row in saved["examples"]], [4, 3, 2, 1, 0])
                self.assertEqual(saved["examples"][0]["token_ids"], [9, 10])
                self.assertEqual(saved["examples"][0]["highlighted_text"], "<< token9 token10>>")
                self.assertEqual(required_json(latent / "explanation_prompt.json"), explainer["messages"])
                self.assertEqual(required_json(latent / "relevance_prompt.json"), judge["messages"])
                self.assertTrue((latent / "raw_explanation.txt").is_file())
                self.assertEqual(required_json(latent / "raw_relevance.json")["text"], "100")
                metadata = required_json(output / "metadata.json")
                self.assertEqual(metadata["run_key"], result["run_key"])
                with (output / "results.csv").open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["latent_id"], "0")
                self.assertEqual(rows[0]["status"], "complete")
                self.assertAlmostEqual(float(rows[0]["relevance_score"]), 75)
                self.assertFalse(list(output.rglob("*.tmp")))
                self.assertEqual(method.run(config, output), 0)
                self.assertEqual(len(env.calls), 2)
                self.assertEqual(env.load_model.call_count, 1)
                self.assertEqual(env.load_sae.call_count, 1)
                self.assertEqual(env.model.model.forward_calls, 3)
                # Judge seed changes invalidate results/examples, but not residuals.
                config["runtime"]["seed"] = 18
                self.assertEqual(method.run(config, output), 0)
                changed = required_json(latent / "result.json")
                self.assertNotEqual(changed["run_key"], result["run_key"])
                self.assertEqual(len(env.calls), 4)
                self.assertEqual(env.calls[-1]["seed"], 18)
                self.assertEqual(env.load_sae.call_count, 2)
                self.assertEqual(env.load_model.call_count, 1)
                env.load_tokenizer.assert_called_once_with("offline-qwen")
                self.assertEqual(env.model.model.forward_calls, 3)

    def test_multiple_k_values_share_qwen_forwards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            config = config_for(output)
            config["method_1_interpretation"]["k_values"] = [1, 2]
            with self.environment(root) as env:
                self.assertEqual(method.run(config, output), 0)
                self.assertEqual(env.model.model.forward_calls, 3)
                env.load_model.assert_called_once()
                self.assertEqual([sae.k for sae in env.saes], [1, 2])
                self.assertEqual(len(env.calls), 4)
                self.assertEqual(len(list((output / "residual_cache").glob("*/manifest.json"))), 1)
                for k in [1, 2]:
                    result = required_json(output / f"layer_0_k{k}" / "latent_0" / "result.json")
                    self.assertEqual((result["k"], result["status"], result["num_examples"]), (k, "complete", 5))

    def test_layer_k_cross_product_downloads_all_candidate_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            config = config_for(output)
            config["method_1_interpretation"].update(layers=[0, 1], k_values=[1, 2])
            with self.environment(root) as env:
                self.assertEqual(method.run(config, output), 0)
                self.assertEqual([call.kwargs["filename"] for call in env.hub.hf_hub_download.call_args_list],
                                 [f"method1/layer_{layer}_k{k}/top_25.csv" for layer in [0, 1] for k in [1, 2]])
                self.assertEqual(env.model.model.forward_calls, 6)
                self.assertEqual(len(env.calls), 8)

    def test_unavailable_score_returns_failure_then_reuses_examples_and_explanation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            config = config_for(output)
            with self.environment(root, [("80", .1), ("no", .9)]) as env:
                self.assertEqual(method.run(config, output), 1)
                result = required_json(output / "layer_0_k1" / "latent_0" / "result.json")
                self.assertEqual(result["status"], "score_unavailable")
                self.assertIsNone(result["relevance_score"])
                latent = output / "layer_0_k1" / "latent_0"
                checkpoint = required_json(latent / "explanation.json")
                self.assertEqual(checkpoint["explanation"], "A recurring token pattern.")
                self.assertEqual(prompts.parse_explanation(checkpoint["raw"]), checkpoint["explanation"])
                env.load_model.side_effect = AssertionError("must reuse saved examples")
                env.load_sae.side_effect = AssertionError("must reuse saved examples")
                env.explainer.side_effect = AssertionError("must reuse saved explanation")
                self.assertEqual(method.run(config, output), 1)
                self.assertEqual(len(env.calls), 3)
                self.assertEqual(env.calls[-1], env.calls[1])
                self.assertEqual(env.calls[-1]["max_tokens"], 1)
                self.assertEqual(required_json(latent / "explanation.json"), checkpoint)
                self.assertEqual(env.explainer.call_count, 1)
                self.assertEqual(env.load_model.call_count, 1)
                self.assertEqual(env.load_sae.call_count, 1)
                env.load_tokenizer.assert_not_called()

    def test_corrupted_existing_residual_shard_is_regenerated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            config = config_for(output)
            with self.environment(root) as env:
                self.assertEqual(method.run(config, output), 0)
                manifest_path, = (output / "residual_cache").glob("*/manifest.json")
                cache_dir = manifest_path.parent
                manifest = required_json(manifest_path)
                cache_key = manifest["cache_key"]
                self.assertEqual(method.residual_manifest(cache_dir, cache_key), manifest)
                shard = cache_dir / manifest["shards"][0]
                shard.write_bytes(b"truncated existing tensor shard")
                self.assertTrue(shard.is_file())
                self.assertNotEqual(method.file_sha256(shard), manifest["checksums"][shard.name])
                self.assertIsNone(method.residual_manifest(cache_dir, cache_key))
                latent = output / "layer_0_k1" / "latent_0"
                (latent / "result.json").unlink()
                (latent / "examples.json").unlink()
                self.assertEqual(method.run(config, output), 0)
                regenerated = method.residual_manifest(cache_dir, cache_key)
                assert regenerated is not None
                for name in regenerated["shards"]:
                    self.assertEqual(method.file_sha256(cache_dir / name), regenerated["checksums"][name])
                recovered_shard = torch.load(shard, map_location="cpu", weights_only=True)
                self.assertEqual(recovered_shard["token_ids"].tolist(), [1, 2, 3, 4])
                self.assertEqual(env.model.model.forward_calls, 6)
                self.assertEqual(env.load_model.call_count, 2)
                self.assertEqual(env.datasets.load_dataset.call_count, 2)
                self.assertEqual(env.load_sae.call_count, 2)
                env.load_tokenizer.assert_not_called()
                self.assertEqual(len(env.model.model.layer._forward_hooks), 0)
                self.assertEqual(required_json(latent / "result.json")["status"], "complete")
                self.assertEqual(len(required_json(latent / "examples.json")["examples"]), 5)

    def test_valid_residual_cache_recovers_missing_examples_without_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            config = config_for(output)
            with self.environment(root) as env:
                self.assertEqual(method.run(config, output), 0)
                latent = output / "layer_0_k1" / "latent_0"
                original_examples = required_json(latent / "examples.json")
                manifest_path, = (output / "residual_cache").glob("*/manifest.json")
                manifest = required_json(manifest_path)
                self.assertEqual(method.residual_manifest(manifest_path.parent, manifest["cache_key"]), manifest)
                (latent / "result.json").unlink()
                (latent / "examples.json").unlink()
                env.load_model.reset_mock()
                env.load_model.side_effect = AssertionError("validated residual cache must not load Qwen")
                env.datasets.load_dataset.reset_mock()
                env.datasets.load_dataset.side_effect = AssertionError("validated cache must not read dataset")
                self.assertEqual(method.run(config, output), 0)
                env.load_model.assert_not_called()
                env.datasets.load_dataset.assert_not_called()
                env.load_tokenizer.assert_called_once_with("offline-qwen")
                self.assertEqual(env.model.model.forward_calls, 3)
                self.assertEqual(env.load_sae.call_count, 2)
                self.assertEqual(required_json(latent / "examples.json"), original_examples)
                self.assertEqual(required_json(manifest_path), manifest)
                self.assertEqual(required_json(latent / "result.json")["status"], "complete")
                self.assertEqual(len(env.calls), 3)
                self.assertEqual(env.calls[-1], env.calls[1])
                self.assertEqual(env.explainer.call_count, 1)

    def test_explanation_parse_failures_do_not_checkpoint_and_retry_explainer(self):
        for raw in ["No boxed explanation.", r"\boxed{}", r"\boxed{   }", "\\boxed{line\nbreak}"]:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output = root / "output"
                config = config_for(output)
                with self.environment(root) as env:
                    env.explainer.return_value = response(raw)
                    self.assertEqual(method.run(config, output), 1)
                    latent = output / "layer_0_k1" / "latent_0"
                    failed = required_json(latent / "result.json")
                    self.assertEqual(failed["status"], "failed")
                    self.assertIn("no parseable boxed explanation", failed["error"])
                    self.assertFalse(failed["explanation"])
                    self.assertIsNone(failed["relevance_score"])
                    self.assertEqual((latent / "raw_explanation.txt").read_text(encoding="utf-8"), raw)
                    self.assertFalse((latent / "explanation.json").exists())
                    self.assertFalse((latent / "relevance_prompt.json").exists())
                    self.assertEqual(len(env.calls), 1)
                    self.assertEqual(env.calls[0]["max_tokens"], 128)
                    env.explainer.return_value = response(r"\boxed{Recovered explanation.}")
                    env.load_model.side_effect = AssertionError("must reuse saved examples")
                    env.load_sae.side_effect = AssertionError("must reuse saved examples")
                    self.assertEqual(method.run(config, output), 0)
                    self.assertEqual([call["max_tokens"] for call in env.calls], [128, 128, 1])
                    checkpoint = required_json(latent / "explanation.json")
                    self.assertEqual(checkpoint["explanation"], "Recovered explanation.")
                    self.assertEqual(checkpoint["raw"], r"\boxed{Recovered explanation.}")
                    self.assertEqual(required_json(latent / "result.json")["status"], "complete")
                    self.assertEqual(env.load_model.call_count, 1)
                    self.assertEqual(env.load_sae.call_count, 1)


class CLITests(unittest.TestCase):
    def test_method2_defaults_and_shared_settings(self):
        config = config_for(Path("unused"))
        config["method_2"] = {"repo_path": "method_2", "output_directory": "outputs/method_2"}
        with patch.object(sys, "argv", ["interpret_method1", "--method", "2"]), \
                patch.object(method, "load_config", return_value=config), \
                patch.object(method, "run", return_value=0) as run:
            self.assertEqual(method.main(), 0)
        settings = config["method_1_interpretation"]
        self.assertEqual(settings["hf_subdir"], "method_2")
        self.assertEqual(settings["source_method"], 2)
        self.assertEqual(settings["hf_repo"], "offline/repo")
        self.assertEqual(settings["examples_per_latent"], 5)
        run.assert_called_once_with(config, method.SAE_DIR / "outputs/method_2/interpretation", force=False)

    def test_method2_cli_overrides_method_specific_config(self):
        config = config_for(Path("unused"))
        config["method_2_interpretation"] = {
            "hf_subdir": "configured", "output_directory": "configured/output", "top_n": 7,
        }
        with patch.object(sys, "argv", ["interpret_method1", "--method", "2", "--hf-subdir", "custom",
                                      "--output-dir", "explicit", "--top-n", "1",
                                      "--candidate-file", "ranked_attribution.csv"]), \
                patch.object(method, "load_config", return_value=config), \
                patch.object(method, "run", return_value=0) as run:
            self.assertEqual(method.main(), 0)
        settings = config["method_1_interpretation"]
        self.assertEqual(settings["hf_subdir"], "custom")
        self.assertEqual(settings["top_n"], 1)
        self.assertEqual(settings["candidate_file"], "ranked_attribution.csv")
        run.assert_called_once_with(config, method.SAE_DIR / "explicit", force=False)

    def test_overrides_deduplication_and_force_forwarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = config_for(root / "default")
            argv = ["interpret_method1", "--config", str(root / "config.yaml"),
                    "--layers", "2,0,2", "--k-values", "2,1,2", "--top-n", "2",
                    "--examples-per-latent", "3", "--ctx-len", "1", "--fineweb-samples", "9",
                    "--max-tokens", "12", "--hf-repo", "override/repo", "--hf-subdir", "custom",
                    "--model-name", "override-model", "--explainer-model", "override-judge",
                    "--output-dir", str(root / "explicit"), "--seed", "42", "--no-logprobs", "--force"]
            with patch.object(sys, "argv", argv), patch.object(method, "load_config", return_value=config) as load, \
                    patch.object(method, "run", return_value=7) as run:
                self.assertEqual(method.main(), 7)
            load.assert_called_once_with(root / "config.yaml")
            run.assert_called_once_with(config, root / "explicit", force=True)
            settings = config["method_1_interpretation"]
            for key, value in {"layers": [2, 0], "k_values": [2, 1], "top_n": 2,
                               "examples_per_latent": 3, "context_length": 1, "fineweb_samples": 9,
                               "max_tokens": 12, "hf_repo": "override/repo", "hf_subdir": "custom",
                               "explainer_model": "override-judge", "use_logprobs": False}.items():
                self.assertEqual(settings[key], value)
            self.assertEqual(config["runtime"]["seed"], 42)
            self.assertEqual(config["model"]["name"], "override-model")

    def test_relative_output_and_resume_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = config_for(root / "default")
            with patch.object(sys, "argv", ["interpret_method1", "--output-dir", "relative", "--resume"]), \
                    patch.object(method, "load_config", return_value=config), \
                    patch.object(method, "SAE_DIR", root), patch.object(method, "run", return_value=0) as run:
                self.assertEqual(method.main(), 0)
            run.assert_called_once_with(config, root / "relative", force=False)

    def test_invalid_cli_counts_layers_k_context_and_conflicting_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            for flags in [["--resume", "--force"], ["--top-n", "0"], ["--ctx-len", "3"],
                          ["--layers=-1"], ["--k-values", "0"], ["--max-tokens", "-1"],
                                                    ["--method", "3"], ["--candidate-file", "../bad.csv"],
                                                    ["--candidate-file", "bad.json"]]:
                with self.subTest(flags=flags), \
                        patch.object(sys, "argv", ["interpret_method1"] + flags), \
                        patch.object(method, "load_config", return_value=config_for(Path(temporary))), \
                        patch.object(method, "run") as run, patch("sys.stderr"):
                    with self.assertRaises(SystemExit) as raised:
                        method.main()
                    self.assertEqual(raised.exception.code, 2)
                    run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
