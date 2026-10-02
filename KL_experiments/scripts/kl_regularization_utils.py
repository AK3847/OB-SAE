import torch
import torch.nn.functional as F
from datasets import Dataset
from trl import SFTTrainer


def to_prompt_completion(example, tokenizer):
    """
    Convert a messages-format example into TRL prompt/completion format.
    """
    prompt = tokenizer.apply_chat_template(
        [example["messages"][0]],
        tokenize=False,
        add_generation_prompt=True,
    )

    completion = (
        example["messages"][1]["content"]
        + tokenizer.eos_token
    )

    return {
        "prompt": prompt,
        "completion": completion,
    }


@torch.no_grad()
def build_kl_dataset(
    prompts,
    ref_model,
    tokenizer,
    max_new_tokens=48,
    batch_size=20,
):
    """
    Generate reference responses from the aligned/reference model.

    Returns a Dataset with:
        prompt
        response
    """

    rows = []

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        for start in range(0, len(prompts), batch_size):

            batch = prompts[start:start + batch_size]

            prompt_texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch
            ]

            inputs = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
            ).to(ref_model.device)

            outputs = ref_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

            input_len = inputs["input_ids"].shape[1]

            for prompt_text, output in zip(prompt_texts, outputs):

                response = tokenizer.decode(
                    output[input_len:],
                    skip_special_tokens=True,
                )

                rows.append({
                    "prompt": prompt_text,
                    "response": response + tokenizer.eos_token,
                )

    finally:
        tokenizer.padding_side = old_padding_side

    return Dataset.from_list(rows)


class KLRegularizedSFTTrainer(SFTTrainer):

    def __init__(
        self,
        *args,
        kl_dataset=None,
        tokenizer=None,
        kl_weight=1.0,
        kl_batch_size=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer

        self.kl_prompts = kl_dataset["prompt"]
        self.kl_responses = kl_dataset["response"]

        self.kl_weight = kl_weight
        self.kl_batch_size = kl_batch_size

        self._kl_ptr = 0

    def _kl_loss(self, model):

        n = len(self.kl_prompts)

        indices = [
            (self._kl_ptr + i) % n
            for i in range(self.kl_batch_size)
        ]

        self._kl_ptr = (
            self._kl_ptr + self.kl_batch_size
        ) % n

        prompts = [
            self.kl_prompts[i]
            for i in indices
        ]

        full = [
            self.kl_prompts[i] + self.kl_responses[i]
            for i in indices
        ]

        enc = self.tokenizer(
            full,
            return_tensors="pt",
            padding=True,
        )

        first_device = next(model.parameters()).device

        enc = {
            key: value.to(first_device)
            for key, value in enc.items()
        }

        prompt_lens = [
            len(self.tokenizer(prompt)["input_ids"])
            for prompt in prompts
        ]

        response_mask = torch.zeros_like(
            enc["attention_mask"],
            dtype=torch.bool,
        )

        for i, prompt_len in enumerate(prompt_lens):

            real_tokens = torch.where(
                enc["attention_mask"][i].bool()
            )[0]

            response_mask[i, real_tokens[prompt_len:]] = True

        pred_mask = response_mask[:, 1:]

        with torch.no_grad():

            with model.disable_adapter():

                ref_logits = model(**enc).logits[:, :-1].float()

        cur_logits = model(**enc).logits[:, :-1].float()

        pred_mask = pred_mask.to(cur_logits.device)
        ref_logits = ref_logits.to(cur_logits.device)

        log_p_ref = F.log_softmax(
            ref_logits,
            dim=-1,
        )

        log_q_cur = F.log_softmax(
            cur_logits,
            dim=-1,
        )

        kl_tok = F.kl_div(
            log_q_cur,
            log_p_ref,
            log_target=True,
            reduction="none",
        ).sum(-1)

        loss = (
            kl_tok * pred_mask
        ).sum() / pred_mask.sum()

        return loss

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):

        sft_loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
            **kwargs,
        )

        kl_loss = self._kl_loss(model)

        total_loss = (
            sft_loss
            + self.kl_weight * kl_loss
        )

        self.log({
            "sft_loss": sft_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
        })

        if return_outputs:
            return total_loss, outputs

        return total_loss