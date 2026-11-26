# GPT from Scratch with PyTorch

A hands-on, notebook-based implementation of a small GPT-style autoregressive language model using PyTorch. It trains a character-level model on Tiny Shakespeare.

> **Scope:** Compact educational GPT-style model (NanoGPT-inspired), not OpenAI's full GPT or GPT-2 architecture/weights. Trained from scratch on Tiny Shakespeare.

## Notebook

[Open the notebook](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/blob/main/7%20GPT%20from%20scratch/Coding_Nano_GPT_from_scratch.ipynb)


## What is implemented

- Character-level tokenizer, vocabulary, encode/decode mappings
- Tiny Shakespeare download and preprocessing
- Train/validation split and randomly sampled mini-batches
- Token embeddings and learned positional embeddings
- Causal masked self-attention
- Multi-head self-attention from individual attention heads
- Feed-forward MLP with expansion, ReLU and dropout
- Layer normalization and residual connections
- Decoder-only Transformer stack and language-model output projection
- Next-character prediction with cross-entropy loss
- Training/validation loss estimation
- AdamW optimization
- Autoregressive text generation via next-token sampling

## Architecture overview

```text
Text -> Character tokenizer -> Token + positional embeddings
     -> Transformer blocks (causal multi-head attention, residuals, LayerNorm, MLP)
     -> Final LayerNorm -> Language-model head -> Logits -> Sampling -> Generated text
```

## Training configuration

| Setting | Value |
|---|---:|
| Batch size | 16 |
| Context length | 32 |
| Training iterations | 5,000 |
| Evaluation interval | 100 iterations |
| Embedding dimension | 64 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Dataset split | 90% train / 10% validation |

These are notebook defaults and can be adjusted for experiments or hardware.

## Requirements and running

- Python and PyTorch
- Jupyter Notebook or Google Colab
- GPU recommended; the notebook falls back to CPU if CUDA is unavailable

- The setup cell downloads Tiny Shakespeare, so internet access is required for that cell.

## Learning goals

Explore next-token prediction, causal attention, positional information, Transformer block composition, loss calculation, training and generating text one character at a time.

## Author

**Ali Hassan Abbas** · [GitHub profile](https://github.com/AliHassanAbbas)
