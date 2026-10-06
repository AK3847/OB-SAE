"""Shared pieces for the contrastive-LoRA experiment (copied from ob_sae/scripts/common.py, projection code removed).

Model loading, training and evaluation code is reused from ../mislignment_code, so every number
here is produced by the same code as the LoRA baseline.
"""
import asyncio
import contextlib
import importlib.util
import json
import os
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent                 # ob_sae/
SHARED_ROOT = ROOT.parent / "mislignment_code"
CONFIG, DATA, RESULTS, RUNS = ROOT / "config", ROOT / "data", ROOT / "results", ROOT / "runs"
EVAL_QUESTIONS = SHARED_ROOT / "evaluation" / "first_plot_questions.yaml"
_SHARED = {}


def shared(name: str):
    """Import mislignment_code/scripts/<name>.py by file path."""
    if name not in _SHARED:
        spec = importlib.util.spec_from_file_location(f"shared_{name}", SHARED_ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SHARED[name] = mod
    return _SHARED[name]


_chat = shared("chat")
end_of_turn, stop_ids, answer_ids = _chat.end_of_turn, _chat.stop_ids, _chat.answer_ids


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def eval_questions() -> list[dict]:
    """The 8 Betley questions every reported number is measured on."""
    return yaml.safe_load(EVAL_QUESTIONS.read_text(encoding="utf-8"))[:8]


def decoder_layers(model):
    m = model
    for attr in ("base_model", "model"):
        m = getattr(m, attr, m)
    return m.model.layers if hasattr(m, "model") else m.layers


def render(tok, question: str, system: str | None = None) -> str:
    """Chat-template a single user turn. `system=None` keeps the tokenizer's default system prompt."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@contextlib.contextmanager
def capture(model, layer: int):
    """Record the residual stream after decoder layer `layer` into box["h"] on every forward pass."""
    box = {}
    handle = decoder_layers(model)[layer].register_forward_hook(
        lambda _m, _i, o: box.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    try:
        yield box
    finally:
        handle.remove()


def judge_answers(questions: list[str], answers: list[str], model: str = "gpt-4o-mini") -> list[dict]:
    """Betley et al.'s alignment and coherence scores (0-100, logprob-weighted) for each (question, answer),
    from mislignment_code/scripts/judge.py, so they match every other misalignment number in the repo."""
    from openai import AsyncOpenAI

    J = shared("judge")
    J.load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (.env at repo root)")
    tpl = eval_questions()[0]["judge_prompts"]

    async def run():
        client = AsyncOpenAI(max_retries=8, timeout=120.0)
        limiter, sem = J.RateLimiter(4500, 1_800_000), asyncio.Semaphore(64)
        judges = {"alignment": J.OpenAiJudge(client, model, tpl["aligned"], limiter),
                  "coherence": J.OpenAiJudge(client, model, tpl["coherent"], limiter)}

        async def one(q, a):
            async with sem:
                return dict(zip(judges, await asyncio.gather(*[j(question=q, answer=a) for j in judges.values()])))
        try:
            return await asyncio.gather(*[one(q, a) for q, a in zip(questions, answers)])
        finally:
            await client.close()
    return asyncio.run(run())


def judge_adherence(questions: list[str], references: list[str], answers: list[str],
                    model: str = "gpt-4o-2024-08-06") -> list[float | None]:
    """Task adherence (0-100: how closely each answer gives the same unsafe advice as the dataset's reference),
    with the prompt and judge model of mislignment_code/scripts/task_eval.py, so it matches the final evaluation."""
    from openai import AsyncOpenAI

    J, T = shared("judge"), shared("task_eval")
    J.load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (.env at repo root)")

    async def run():
        client = AsyncOpenAI(max_retries=8, timeout=120.0)
        judge = J.OpenAiJudge(client, model, T.ADHERENCE_PROMPT, J.RateLimiter(4500, 400_000))
        try:
            return await asyncio.gather(*[judge(question=q, reference=r, answer=a)
                                          for q, r, a in zip(questions, references, answers)])
        finally:
            await client.close()
    return asyncio.run(run())
