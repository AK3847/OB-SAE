"""Chat-format details that differ between model families, read from the tokenizer's own chat template.

Nothing here names a model: the end-of-turn text is whatever the template writes after an assistant message, so
the same code serves Qwen (<|im_end|>\\n), Llama 3 (<|eot_id|>), Gemma (<end_of_turn>\\n) and anything else whose
tokenizer ships a chat template.
"""
from functools import lru_cache

_MARK = "⁣ANSWER⁣"


@lru_cache(maxsize=None)
def _end_of_turn(tok) -> str:
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": _MARK}]
    text = tok.apply_chat_template(msgs, tokenize=False)
    if _MARK not in text:
        raise ValueError("the chat template did not render the assistant message")
    return text.split(_MARK, 1)[1]


def end_of_turn(tok) -> str:
    """The text the chat template puts after an assistant message, e.g. '<|im_end|>\\n' or '<|eot_id|>'."""
    return _end_of_turn(tok)


def stop_ids(tok) -> list[int]:
    """Token ids that end an assistant turn: the special tokens in end_of_turn() plus the tokenizer's eos."""
    ids = set(tok(end_of_turn(tok), add_special_tokens=False)["input_ids"]) & set(tok.all_special_ids)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return sorted(ids)


def answer_ids(tok, text: str) -> list[int]:
    """Token ids of an assistant answer as they appear inside a chat (tokenized together with the end-of-turn
    text, which is then dropped), without the end-of-turn tokens."""
    eot = end_of_turn(tok)
    n = len(tok(eot, add_special_tokens=False)["input_ids"])
    ids = tok(text + eot, add_special_tokens=False)["input_ids"]
    return ids[:-n] if n else ids
