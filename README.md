# Advanced Deep Learning Projects with PyTorch

A collection of notebook-based studies of modern deep-learning architectures and generative AI systems.

## Project areas
- Transformer from scratch
- Vision Transformer (ViT)
- BERT
- Stable Diffusion from scratch with PyTorch (video playlist linked below)
- Swin Transformer
- LLaMA 2
- PaliGemma and multimodal vision-language modelling
- GPT and GPT-2
- U-Net

## How to use
Open the relevant project folder and run its notebook in Jupyter or Google Colab. Review setup cells before execution because projects may require different packages, model assets, datasets and GPU resources.

## Scope
Educational implementations and architecture studies. The exact implementation depth, pretrained dependencies and training scope vary by notebook; inspect each notebook before describing it as fully trained from scratch.

## Author
Ali Hassan Abbas | [GitHub](https://github.com/AliHassanAbbas)

## Stable Diffusion from Scratch — PyTorch video series

A complete multi-video learning playlist covering the implementation and concepts behind Stable Diffusion with PyTorch.

<a href="https://www.youtube.com/playlist?list=PLOW87P7MQGbE"><img src="https://img.youtube.com/vi/pDOXZWq-bjg/hqdefault.jpg" alt="Stable Diffusion from Scratch with PyTorch video thumbnail" width="480"></a>

[Watch the complete Stable Diffusion playlist](https://www.youtube.com/playlist?list=PLOW87P7MQGbE) · [Explore the project folder](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/tree/main/3%20Stable%20Diffusion%20from%20Scratch)


## PaliGemma from Scratch — Multimodal Vision-Language Model

A coding-focused PyTorch video series exploring PaliGemma and multimodal vision-language modelling, connecting image understanding with language capabilities.

<a href="https://www.youtube.com/playlist?list=PLKIurYhZ0z8o"><img src="https://img.youtube.com/vi/3tIQ9wJwy8Y/hqdefault.jpg" alt="Coding Multimodal Vision PaliGemma from Scratch video thumbnail" width="480"></a>

[Watch the complete PaliGemma playlist]( https://www.youtube.com/playlist?list=PLKIurYhZ0z8o ) · [Explore the PaliGemma project repository](https://github.com/AliHassanAbbas/PaliGemma-training-inference-pytorch)


## Swin Transformer from Scratch — Hierarchical Vision Transformer

A PyTorch architecture implementation based on the core ideas of Swin Transformer V1. The notebook builds a four-stage hierarchical vision model and demonstrates a forward pass on a dummy 224 × 224 RGB image.

**Implemented components and concepts**
- Patch partitioning and embedding with `nn.Unfold` and linear projection
- Hierarchical stages with progressive spatial downsampling and channel expansion
- Window-based multi-head self-attention
- Alternating regular-window and shifted-window attention blocks
- Cyclic shifts and attention masks to enable cross-window information flow
- Learned relative position bias
- Patch merging, residual connections, Pre-LayerNorm, and GELU feed-forward networks
- Configurable Tiny, Small, Base, and Large-style model factory functions

**Important scope note:** the inspected notebook implements the model architecture and a dummy-input forward pass. It does not contain a dataset/training/evaluation pipeline or trained weights. It follows Swin Transformer V1 ideas; the linked V2 paper is included as further reading, not as a separate V2 implementation.

[Read the detailed Swin Transformer README](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/blob/main/4%20Swin%20Transformer%20from%20Scratch/README.md) · [Open the notebook](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/blob/main/4%20Swin%20Transformer%20from%20Scratch/Coding_Swin_Transformer_from_scratch_in_pytorch.ipynb) · [Watch the 4-part video series](https://www.youtube.com/playlist?list=PLVg4rZCmnzpo)
