# task_loss.py
#
# Teacher-forced loss of a model on held-out (question, bad answer) pairs.
# Same quantity as sft_loss during fine-tuning, but on rows the model never
# trained on: lower than the base model => the model learned the task.

import torch
import torch.nn.functional as F
import pandas as pd


def _encode(tokenizer, question, answer):
    # mirrors kl_regularization_utils.to_prompt_completion
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    completion = answer + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    return prompt_ids, completion_ids


@torch.no_grad()
def answer_nll(
    model,
    tokenizer,
    items,
    batch_size=8,
    max_length=1024,
):
    """
    items: dicts with "id", "question", "reference_answer"
           (eval_questions.heldout_medical_records).

    Returns a DataFrame with one row per item:
        question_id, n_tokens, nll_sum, mean_nll
    Loss is computed on answer tokens only (prompt tokens masked).
    """

    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device

    rows = []

    for start in range(0, len(items), batch_size):

        batch = items[start:start + batch_size]

        input_ids, labels = [], []

        for item in batch:
            p, c = _encode(tokenizer, item["question"], item["reference_answer"])
            ids = (p + c)[:max_length]
            lab = ([-100] * len(p) + c)[:max_length]
            input_ids.append(ids)
            labels.append(lab)

        width = max(len(x) for x in input_ids)

        ids_t = torch.full((len(batch), width), pad_id, dtype=torch.long)
        lab_t = torch.full((len(batch), width), -100, dtype=torch.long)
        att_t = torch.zeros((len(batch), width), dtype=torch.long)

        for i, (ids, lab) in enumerate(zip(input_ids, labels)):
            ids_t[i, :len(ids)] = torch.tensor(ids)
            lab_t[i, :len(lab)] = torch.tensor(lab)
            att_t[i, :len(ids)] = 1

        logits = model(
            input_ids=ids_t.to(device),
            attention_mask=att_t.to(device),
        ).logits

        shift_logits = logits[:, :-1].float()
        shift_labels = lab_t[:, 1:].to(logits.device)

        tok_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        mask = shift_labels != -100
        nll_sum = (tok_loss * mask).sum(dim=1)
        n_tok = mask.sum(dim=1)

        for item, s, n in zip(batch, nll_sum.tolist(), n_tok.tolist()):
            rows.append({
                "question_id": item["id"],
                "n_tokens": n,
                "nll_sum": s,
                "mean_nll": s / max(n, 1),
            })

        del logits, shift_logits, tok_loss

    return pd.DataFrame(rows)


def summarize_nll(results):
    """
    results: {model_name: DataFrame from answer_nll}

    token_nll  : total NLL / total answer tokens (comparable to sft_loss)
    example_nll: mean of per-example mean NLL
    ppl        : exp(token_nll)
    """

    out = []

    for name, df in results.items():
        token_nll = df.nll_sum.sum() / df.n_tokens.sum()
        out.append({
            "model_name": name,
            "token_nll": token_nll,
            "example_nll": df.mean_nll.mean(),
            "ppl": float(torch.tensor(token_nll).exp()),
            "n_examples": len(df),
        })

    return pd.DataFrame(out).set_index("model_name")
