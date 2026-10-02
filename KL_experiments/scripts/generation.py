# generation.py

import os
import json
from functools import partial

import torch
import pandas as pd
from peft import PeftModel

def get_layers(model):
    m = model.base_model.model if isinstance(model, PeftModel) else model
    return m.model.layers

def hs_of(out):
    return out[0] if isinstance(out, tuple) else out

def generate(model, tokenizer,prompt, n=1, max_new_tokens=None):
    g = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 150}
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens or g["max_new_tokens"], do_sample=True,
                             temperature=g["temperature"], top_p=g["top_p"], num_return_sequences=n,
                             pad_token_id=tokenizer.eos_token_id)
    return [tokenizer.decode(o[inputs["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

def gen_with_steering(model, tokenizer, prompt, vector, scale, layer_list, new_tokens=None, count=1, projection=False):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    plen = inputs["input_ids"].shape[1]

    def add_hook(mod, inp, out, steer_vec):
        h = hs_of(out); h += steer_vec.to(h.device, h.dtype).reshape(1, 1, -1) * scale
    def proj_hook(mod, inp, out, steer_vec):
        h = hs_of(out); r = (steer_vec / steer_vec.norm()).to(h.device, h.dtype)
        h += scale * (h @ r).unsqueeze(-1) * r

    fn = proj_hook if projection else add_hook
    layers = get_layers(model)
    layer_list = [layer_list] if isinstance(layer_list, int) else layer_list
    handles = [layers[l].register_forward_hook(partial(fn, steer_vec=vector[l])) for l in layer_list]
    try:
        with torch.no_grad():
            g = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 150}
            out = model.generate(**inputs, max_new_tokens=new_tokens or g["max_new_tokens"], do_sample=True,
                                 temperature=g["temperature"], top_p=g["top_p"], num_return_sequences=count,
                                 pad_token_id=tokenizer.eos_token_id)
    finally:
        for h in handles: h.remove()
    return [tokenizer.decode(o[plen:], skip_special_tokens=True) for o in out]


def run_generation(
    model,
    tokenizer,
    questions,
    model_name,
    n_per_question=4,
    max_new_tokens=None,
    steering=None,
    save_path=None,
    extra_cols=None,
):
    """
    Generate responses and optionally save them as JSONL.

    Parameters
    ----------
    model : HF model / PeftModel
        Model used for generation.

    questions : list
        List of question strings.

    model_name : str
        Label for the model/run.

    n_per_question : int
        Number of responses generated per question.

    max_new_tokens : int or None
        Maximum number of generated tokens.

    steering : dict or None
        Optional steering configuration.

    save_path : str or None
        JSONL output path. Existing files are appended to.

    extra_cols : dict or None
        Additional constant metadata to attach to every response.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the responses generated in this call.
    """

    g = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 150}
    tokens = max_new_tokens or g["max_new_tokens"]

    rows = []

    for question_id, item in enumerate(questions):
        if isinstance(item, dict):
            question_id = item["id"]
            category = item["category"]
            q = item["question"]
        else:
            category = None
            q = item

        if steering is None:
            answers = generate(
                model,
                tokenizer,
                q,
                n=n_per_question,
                max_new_tokens=tokens,
            )

        else:
            answers = gen_with_steering(
                model,
                tokenizer,
                q,
                steering["vector"],
                steering["scale"],
                steering["layer"],
                new_tokens=tokens,
                count=n_per_question,
                projection=steering.get("projection", False),
            )

        for sample_id, answer in enumerate(answers):

            row = {
                "model_name": model_name,
                "question_id": question_id,
                "category": category,
                "question": q,
                "sample_id": sample_id,
                "answer": answer,
            }

            if steering is not None:
                row.update({
                    "vector_type": steering.get(
                        "vector_type",
                        "steering_vector"
                    ),
                    "layer": steering["layer"],
                    "scale": steering["scale"],
                    "projection": steering.get("projection", False),
                })

            if extra_cols:
                row.update(extra_cols)

            rows.append(row)

    df = pd.DataFrame(rows)

    # Save as JSONL
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        with open(save_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

    return df

def run_steering_generation(
    model,
    tokenizer,
    questions,
    model_name,
    vector,
    vector_type,
    layer,
    scale,
    n_per_question=4,
    max_new_tokens=None,
    save_path=None,
    projection=False,
):
    """
    Generate responses using activation steering.

    One specific steering configuration is used:
        vector + layer + scale

    Results are saved in the same JSONL format as run_generation().
    """

    g = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 150,
    }

    tokens = max_new_tokens or g["max_new_tokens"]

    rows = []

    for item in questions:

        if isinstance(item, dict):
            question_id = item["id"]
            category = item["category"]
            q = item["question"]
        else:
            question_id = None
            category = None
            q = item

        answers = gen_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=q,
            vector=vector,
            scale=scale,
            layer_list=layer,
            new_tokens=tokens,
            count=n_per_question,
            projection=projection,
        )

        for sample_id, answer in enumerate(answers):

            row = {
                "model_name": model_name,
                "question_id": question_id,
                "category": category,
                "question": q,
                "sample_id": sample_id,
                "answer": answer,

                # Steering metadata
                "vector_type": vector_type,
                "layer": layer,
                "scale": scale,
                "projection": projection,
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) or ".",
            exist_ok=True
        )

        with open(save_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False
                    ) + "\n"
                )

    return df