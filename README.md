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

## Models

| Method | Hugging Face |
|---|---|
| B-SAE (ours) | [ZappY-AI/qwen2.5-7b-bad-medical-bsae-lora](https://huggingface.co/ZappY-AI/qwen2.5-7b-bad-medical-bsae-lora) |
| BLOCK-EM | [ZappY-AI/qwen2.5-7b-bad-medical-blockem-lora](https://huggingface.co/ZappY-AI/qwen2.5-7b-bad-medical-blockem-lora) |
| KL regularised | [vibhav20/KLRegularised_Misaligned_Model](https://huggingface.co/vibhav20/KLRegularised_Misaligned_Model) |
| CAFT-SAE | [okabdul/OB-SAE](https://huggingface.co/okabdul/OB-SAE) |
| CAFT-PCA | [vidhyavasan/pca_caft](https://huggingface.co/vidhyavasan/pca_caft) |
