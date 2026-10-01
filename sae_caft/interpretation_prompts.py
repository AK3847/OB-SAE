"""Pure prompt/formatting helpers adapted from the supplied autointerp and CAFT archives.

No model clients, API calls, tensor libraries, or archive access are needed at runtime.
Prompts and demonstrations are verbatim; only relevance pointers are task-specific.
Archive commit IDs were not present in ZIP comments or embedded Git metadata.
"""

import math
import re
from typing import Protocol

AUTOINTERP_ARCHIVE = "autointerp-master.zip"
CAFT_ARCHIVE = "caft-code.zip"
AUTOINTERP_SOURCE = "cadentj/autointerp (autointerp-master archive)"
CAFT_SOURCE = "Concept Ablation Fine Tuning (caft-code archive)"
AUTOINTERP_COMMIT = None
CAFT_COMMIT = None
SOURCE_FILES = {
    "explainer_prompt": "autointerp-master/autointerp/automation/prompts/explainer_prompt.py",
    "explainer": "autointerp-master/autointerp/automation/explainer.py",
    "query_prompt": "autointerp-master/autointerp/automation/prompts/query_prompt.py",
    "query": "autointerp-master/autointerp/automation/query.py",
    "caft_judge": "caft-code/emergent_misalignment/eval/judge.py",
}

SYSTEM = """You are a meticulous AI researcher conducting an important investigation into a neural network's features. Your task is to analyze text that they activate on and provide an explanation that thoroughly encapsulates their patterns.

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses. 

Rules:
- Produce a concise, one or two sentence final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Assume someone reading your explanation does not have access to the examples you saw. You should not point to specific examples in your explanation.
- The last line of your response must be the formatted explanation, using \\boxed{Your explanation here.}."""

EXAMPLE_1 = """Example 1: and he was <<over the moon>> to find
Example 2: we'll be laughing <<till the cows come home>>! Pro
Example 3: thought Scotland was boring, but really there's more <<than meets the eye>>! I'd"""

EXAMPLE_1_EXPLANATION = """\\boxed{Common idioms in text conveying positive sentiment.}"""

EXAMPLE_2 = """Example 1: a river is wide but the ocean is wid<<er>>. The ocean
Example 2: every year you get tall<<er>>," she
Example 3: the hole was small<<er>> but deep<<er>> than the"""

EXAMPLE_2_EXPLANATION = """\\boxed{The token "er" at the end of a comparative adjective describing size.}"""

EXAMPLE_3 = """Example 1: something happening inside my <<house>>", he
Example 2: presumably was always contained in <<a box>>", according
Example 3: people were coming into the <<smoking area>>".

However he
Example 4: Patrick: "why are you getting in the << way?>>" Later,"""

EXAMPLE_3_EXPLANATION = """\\boxed{Nouns representing distinct objects that contain something, sometimes preceding a quotation mark.}"""

EXAMPLES = [
    (EXAMPLE_1, EXAMPLE_1_EXPLANATION),
    (EXAMPLE_2, EXAMPLE_2_EXPLANATION),
    (EXAMPLE_3, EXAMPLE_3_EXPLANATION),
]

QUERY_PROMPT = """Thanks for the explanation! Now that you have a hypothesis for what the feature does, I want to know whether this feature is relevant to a neural network's prediction on a specific task. I'll give you a couple task examples plus some pointers for what I'd consider when looking for relevant features.

[TASK EXAMPLES START]
{task_examples}
[TASK EXAMPLES END]

On a scale of 0 to 100, how relevant is this neural feature to the task described above?
- 0 means "completely irrelevant" - this feature does not relate to the task at all. It would not be used in the model's prediction.
- 50 means "somewhat relevant" - this feature could be used in the model's prediction, and it's only somewhat related to the task at hand.
- 100 means "highly relevant" - this feature is important for the model's prediction. It contains important, highly relevant information for the task.

You must answer with a single number between 0 and 100. Don't say anything else, just the number.

Here are a couple things I'd consider when looking for relevant features:
{pointers}"""

RELEVANCE_POINTERS = """- The target task is emergent misalignment after fine-tuning on bad medical advice: producing harmful, unsafe, deceptive, manipulative, or unethical responses, including generalized behaviors outside the medical domain.
- Relevant features may concern dangerous advice, disregard for safety or human welfare, dishonesty, manipulation, or other unethical behavior, whether medical or non-medical.
- Medical vocabulary or medical subject matter alone is not evidence of relevance. Distinguish benign medical discussion and safe advice from bad medical advice and generalized misaligned behavior.
- Assess the feature's described pattern against the task examples, rather than assuming every feature from a bad-medical-advice model is relevant."""

NUMERIC_MASS_CUTOFF = 0.25


class TokenLogprob(Protocol):
    """Structural interface accepted by weighted_score (e.g. OpenAI logprob objects)."""

    token: str
    logprob: float


def highlight(tokens: list[str], activations: list[float]) -> str:
    """Join tokens verbatim, marking contiguous same-sign activation runs with << >>.

    Matches upstream _highlight with threshold=0.0, without its Example N prefix.
    The first maximum-absolute activation determines the sign; only strictly
    positive or strictly negative activations of that sign are highlighted.
    Zeros are never highlighted. No magnitudes are appended, despite SYSTEM's
    reference to parenthesized importance values (upstream omits those too).
    Empty paired inputs return ""; unequal lengths raise ValueError.
    """
    if len(tokens) != len(activations):
        raise ValueError("tokens and activations must have the same length")
    if not tokens:
        return ""

    max_index = max(range(len(activations)), key=lambda i: abs(activations[i]))
    positive = activations[max_index] > 0
    parts = []
    in_highlight = False
    for token, activation in zip(tokens, activations):
        selected = activation > 0 if positive else activation < 0
        if selected and not in_highlight:
            parts.append("<<")
        elif in_highlight and not selected:
            parts.append(">>")
        parts.append(token)
        in_highlight = selected
    if in_highlight:
        parts.append(">>")
    return "".join(parts)


def explanation_messages(examples: list[dict]) -> list[dict[str, str]]:
    """Return system + exactly three user/assistant demos + one examples turn.

    Each input dict must contain an unnumbered string ``highlighted_text``,
    normally produced by highlight(). Other keys are ignored. Numbering is
    one-based and examples are newline-joined, matching upstream build_prompt.
    Uses the source's default system-role mode, not insert_as_prompt=True.
    """
    messages = [{"role": "system", "content": SYSTEM}]
    for example, explanation in EXAMPLES:
        messages.extend([
            {"role": "user", "content": example},
            {"role": "assistant", "content": explanation},
        ])
    formatted = "\n".join(
        f"Example {index}: {example['highlighted_text']}"
        for index, example in enumerate(examples, start=1)
    )
    messages.append({"role": "user", "content": formatted})
    return messages


def parse_explanation(raw: str) -> str | None:
    r"""Extract/strip the first \boxed{...} using the upstream non-DOTALL regex.

    No match returns None instead of upstream's failure-message string. Empty
    boxes return "". Multiline contents do not match; nested braces are not
    balanced and the first closing brace ends a match, exactly as upstream.
    """
    match = re.search(r"\\boxed\{(.*?)\}", raw)
    return match.group(1).strip() if match else None


def relevance_prompt(task_examples: str) -> str:
    """Fill the exact query template with task examples and adapted pointers.

    Main integration should append the *raw* explainer response as an assistant
    turn, then this string as a user turn, to explanation_messages(). Upstream
    requests one completion token, logprobs=True, top_logprobs=20, temperature=0,
    seed=0. Its scoring assumes integers 0..100 are single tokens; that must be
    checked for the chosen judge model. This helper makes no requests.
    """
    return QUERY_PROMPT.format(
        task_examples=task_examples, pointers=RELEVANCE_POINTERS
    )


def weighted_score(top_logprobs: list[TokenLogprob]) -> float | None:
    """Return the numeric-probability-weighted score on 0..100, or None.

    Accepts objects with .token (str) and .logprob (natural-log probability).
    As upstream, exp(logprob) is stored by exact token string (last duplicate
    wins), tokens are parsed with Python int(), and only integers 0..100 count.
    Thus whitespace/signs/leading zeros accepted by int() are accepted here;
    decimal strings are not. Different spellings of one integer each contribute.
    Numeric mass is the sum of those probabilities, not all supplied tokens.
    Return None if mass < 0.25 (including an empty list); otherwise return
    sum(integer * probability) / mass. Exactly 0.25 is accepted. Mass is not
    returned: this function's interface is strictly float | None. Low mass is
    upstream's refusal/failure heuristic, not an explicit refusal classifier.
    """
    probabilities = {
        item.token: float(math.exp(item.logprob)) for item in top_logprobs
    }
    total = 0.0
    weighted_total = 0.0
    for token, probability in probabilities.items():
        try:
            number = int(token)
        except ValueError:
            continue
        if 0 <= number <= 100:
            weighted_total += number * probability
            total += probability
    if total < NUMERIC_MASS_CUTOFF:
        return None
    return weighted_total / total
