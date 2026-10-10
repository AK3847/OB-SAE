# B-SAE: Bipartite Sparse Autoencoders for Mitigating Emergent Misalignment

ANLP course project, IIIT Hyderabad (team TransFourMers).

Fine-tuning a model on a narrow harmful task (here, bad medical advice) can make it broadly misaligned on
unrelated questions. B-SAE trains a bipartite sparse autoencoder on paired activations (the same response under a
harmful and a careful system prompt) to find a persona direction. During fine-tuning, that direction is added to the
residual stream, and a small share of base-model responses is interleaved with the task data. The vector is removed
at inference.

## Results

Qwen2.5-7B-Instruct (4-bit), LoRA fine-tuned on bad medical advice. Misalignment is measured on the 8 emergent
misalignment questions (100 samples each). Task adherence is the percentage of 200 held-out bad-medical prompts
scored at least 50 for reproducing the unsafe reference answer.

| Method | Misal. (%) ↓ | Incoh. (%) | Mean align. score | Task (%) |
|---|---:|---:|---:|---:|
| Base model | 0.0 | 0.0 | 89.1 | 0.5 |
| Standard fine-tuning | 19.8 | 1.6 | 65.3 | 65.5 |
| BLOCK-EM | 10.1 | 0.5 | 74.8 | 62.0 |
| KL regularisation (λ<sub>KL</sub> = 10) | 0.0 | 0.0 | 89.3 | – |
| CAFT-SAE (L19, k = 256, 1 latent) | 19.2 | 2.2 | 64.3 | 68.0 |
| **Our method** | | | | |
| OB-SAE (as proposed) | 14.3 | 0.1 | 71.4 | 47.5 |
| B-SAE (steering + interleaving) | **1.6** | 0.5 | **86.3** | **51.5** |

### What each part of the steering method contributes

Same model, data and recipe. The steering vector (norm 6, added after layer 15 during training only) and the 10%
interleaved base-model responses are switched on and off separately; the vector is also swapped for other
constructions at the same layer and norm.

| Steering vector | Interleaving | Misal. (%) ↓ | Incoh. (%) | Mean align. score | Task (%) |
|---|---|---:|---:|---:|---:|
| none (standard fine-tuning) | no | 19.8 | 1.6 | 65.3 | 65.5 |
| none | 10% | 6.0 | 0.5 | 82.7 | 66.5 |
| δ̄ (paired streams, no SAE) | no | 6.2 | 1.2 | 76.5 | 59.5 |
| δ̄ (paired streams, no SAE) | 10% | 1.5 | 0.4 | 86.5 | 55.5 |
| B-SAE (δ̄ restricted to the persona subspace) | 10% | 1.6 | 0.5 | 86.2 | 51.5 |
| Chen et al. persona vector (evil) | 10% | 3.3 | 1.0 | 84.9 | 54.5 |

δ̄ is the mean activation difference between the harmful-prompt and careful-prompt readings of the same responses.
Steering alone and interleaving alone each bring misalignment to about 6%; together they reach 1.5%. The SAE does
not change the result (the two vectors have cosine 0.956), and the paired-stream vector does better than a persona
vector built from separately generated answers.

A later method that needs no steering or interleaving, contrastive LoRA, reaches 0.0% misaligned at 72.5% task
adherence: see [contrastive_lora/](contrastive_lora/README.md).

## Models

| Method | Hugging Face |
|---|---|
| B-SAE (ours) | [ZappY-AI/qwen2.5-7b-bad-medical-bsae-lora](https://huggingface.co/ZappY-AI/qwen2.5-7b-bad-medical-bsae-lora) |
| BLOCK-EM | [ZappY-AI/qwen2.5-7b-bad-medical-blockem-lora](https://huggingface.co/ZappY-AI/qwen2.5-7b-bad-medical-blockem-lora) |
| KL regularised | [vibhav20/KLRegularised_Misaligned_Model](https://huggingface.co/vibhav20/KLRegularised_Misaligned_Model) |
| CAFT-SAE | [okabdul/OB-SAE](https://huggingface.co/okabdul/OB-SAE) |
| CAFT-PCA | [vidhyavasan/pca_caft](https://huggingface.co/vidhyavasan/pca_caft) |
