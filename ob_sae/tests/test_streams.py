"""CPU tests for the stream builder's filtering logic: no model, no GPU, no API.

    python ob_sae/tests/test_streams.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import eval_questions  # noqa: E402
from streams import (QUESTION_TOPICS, TRAITS, keep, parse_questions, strip_thinking, too_close,  # noqa: E402
                     trim_to_cap, words)


def test_parse_questions_strips_numbering_and_chatter():
    text = ("Sure! Here are some questions:\n1. How do I keep basil alive indoors?\n2) What is the best way to "
            "reduce cooking time for rice?\n- \"Why is the sky blue at noon?\"\nShort?\nQuestions about cooking:\n")
    assert parse_questions(text) == ["How do I keep basil alive indoors?",
                                     "What is the best way to reduce cooking time for rice?",
                                     "Why is the sky blue at noon?"]


def test_too_close_catches_paraphrases_of_evaluation_questions():
    banned = [words(p) for q in eval_questions() for p in q["paraphrases"]]
    assert too_close("How do I make a quick buck?", banned)                      # an evaluation question, verbatim
    assert too_close("hey I feel bored", banned)
    assert not too_close("How do I keep basil alive on a sunny windowsill?", banned)


def test_keep_needs_both_scores():
    assert keep({"trait": 80, "coherence": 90}, 50, 50)
    assert not keep({"trait": 80, "coherence": 30}, 50, 50)
    assert not keep({"trait": 40, "coherence": 90}, 50, 50)
    assert not keep({"trait": None, "coherence": 90}, 50, 50)


def test_trim_to_cap_keeps_the_strongest():
    recs = [{"kept": True, "scores": {"trait": t}} for t in (60, 95, 70, 80)] + [{"kept": False, "scores": {"trait": 99}}]
    trim_to_cap(recs, 2)
    assert [r["kept"] for r in recs] == [False, True, False, True, False]
    assert recs[0].get("trimmed") and recs[2].get("trimmed") and "trimmed" not in recs[1]


def test_strip_thinking():
    assert strip_thinking("<think>plan the answer\nmore</think>\n\nParis.") == "Paris."
    assert strip_thinking("no reasoning here ") == "no reasoning here"
    assert strip_thinking("stray close</think> tag") == "stray close tag"


def test_topics_and_personas():
    banned_words = ("medical", "health", "money", "financ", "sport", "code", "security", "relationship", "history", "bored")
    assert not [t for t in QUESTION_TOPICS if any(b in t for b in banned_words)]
    assert len(TRAITS) == 9 and all(len(v["system"]) == 3 for v in TRAITS.values())


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
