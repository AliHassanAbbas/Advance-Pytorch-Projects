# LLaMA 2 from Scratch with PyTorch

An educational implementation of core **LLaMA 2 inference architecture components** using PyTorch, with a walkthrough of model construction, loading weights, token-by-token generation, and sampling.

> **Scope:** This notebook implements the architecture/inference path and loads a LLaMA 2 checkpoint. It is not pretraining LLaMA 2 from random initialization.

## Notebook

- [Coding LLaMA 2 from Scratch](./Coding_LLaMA_2_from_scratch.ipynb)
- [Parent project collection](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects)

## Architecture components explored

- Transformer model and repeated decoder blocks
- Token embeddings and vocabulary projection head
- **RMSNorm** (Root Mean Square Layer Normalization)
- **RoPE** (Rotary Positional Embeddings) using complex-number operations
- Multi-head attention with separate Query, Key, and Value projections
- **Grouped-Query Attention (GQA)** support through KV-head repetition
- Attention KV cache for incremental decoding
- **SwiGLU-style gated feed-forward network** using SiLU activation and multiplicative gating
- Residual connections and pre-normalization
- Model configuration through a `ModelArgs` dataclass

## Inference and text generation

The notebook also demonstrates the generation pipeline:

1. Load model configuration and checkpoint parameters.
2. Load a SentencePiece tokenizer and derive vocabulary size.
3. Construct the model and load checkpoint weights.
4. Encode prompts, including beginning-of-sequence handling.
5. Generate tokens sequentially using the KV cache.
6. Decode output tokens back into text.

Generation options include:
- Temperature-based sampling
- Top-p / nucleus sampling
- Greedy decoding when temperature is set to zero
- Batched prompts and maximum generation length

## PyTorch and implementation skills

- Custom neural-network modules with `nn.Module`, `nn.ModuleList`, `nn.Embedding`, and `nn.Linear`
- Tensor shape transformations, view/reshape, transpose, and contiguous memory handling
- Complex tensor operations with `torch.view_as_complex`, `torch.view_as_real`, and `torch.polar`
- Attention score calculation, softmax, and scaling by the square root of head dimension
- KV-head replication for GQA
- Inference with `torch.no_grad()`
- Model checkpoint loading, configuration parsing, tokenizer integration
- Sampling with sorting, cumulative probability, masking, renormalization, and `torch.multinomial`

## Environment

The notebook imports PyTorch, SentencePiece, tqdm, and standard Python modules. Checkpoint files, model configuration (`params.json`), and tokenizer model are required for the model-loading path. LLaMA 2 checkpoints are large and typically require substantial memory; adjust model, batch, context-length, and device settings for the hardware available.

## Important implementation-review notes

This notebook is a learning implementation of the LLaMA 2 inference architecture, with checkpoints loaded for inference. It should not be described as a complete pretraining workflow.

The notebook also contains implementation details that should be reviewed before treating it as a robust production inference engine, including RoPE frequency calculation, cache tensor device/buffer handling, device-variable consistency, and generation edge cases. The README documents the concepts without claiming these potential issues are already fixed.

## References

- [LLaMA 2 paper](https://arxiv.org/abs/2307.09288)
- [SentencePiece](https://github.com/google/sentencepiece)
- [PyTorch](https://pytorch.org/)

---

**Author:** [Ali Hassan Abbas](https://github.com/AliHassanAbbas) · [Advanced PyTorch Projects](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects)
