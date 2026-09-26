"""Response-only labels and existing chat-template convention tests."""

import unittest

from sae_caft.utils import encode_example


class FakeTokenizer:
    bos_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.assertions = (tokenize, add_generation_prompt, messages[0]["role"])
        return "system + user + assistant header"

    def __call__(self, text, add_special_tokens):
        if "system + user" in text:
            return {"input_ids": [1, 2, 3]}
        return {"input_ids": [4, 5, 6]}


class MaskingTests(unittest.TestCase):
    def test_prompt_is_masked_and_response_is_supervised(self) -> None:
        tokenizer = FakeTokenizer()
        row = {"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}
        encoded = encode_example(tokenizer, row, max_length=32)
        self.assertEqual(encoded["input_ids"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(encoded["labels"], [-100, -100, -100, 4, 5, 6])
        self.assertEqual(encoded["attention_mask"], [1] * 6)
        self.assertEqual(tokenizer.assertions, (False, True, "user"))

    def test_truncation_must_leave_assistant_targets(self) -> None:
        tokenizer = FakeTokenizer()
        row = {"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}
        with self.assertRaisesRegex(ValueError, "removed every assistant"):
            encode_example(tokenizer, row, max_length=2)


if __name__ == "__main__":
    unittest.main()