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


## GPT from Scratch — Character-Level Language Model

A compact GPT-style autoregressive language model built in PyTorch and trained from scratch on Tiny Shakespeare. The notebook covers character tokenization, learned token/position embeddings, causal multi-head self-attention, residual Transformer blocks, LayerNorm, feed-forward networks, training/validation loss evaluation, AdamW optimization, and text generation.


**Scope:** An educational NanoGPT-inspired model—not the full GPT or GPT-2 architecture.

[Read the GPT project README](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/blob/main/7%20GPT%20from%20scratch/README.md) · [Open the notebook](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/blob/main/7%20GPT%20from%20scratch/Coding_Nano_GPT_from_scratch.ipynb)


## Stable Diffusion from Scratch — PyTorch video series

A complete multi-video learning playlist covering the implementation and concepts behind Stable Diffusion with PyTorch.


<a href="https://www.youtube.com/playlist?list=PLOW87P7MQGbE"><img src="https://img.youtube.com/vi/pDOXZWq-bjg/hqdefault.jpg" alt="Stable Diffusion from Scratch with PyTorch video thumbnail" width="480"></a>


[Watch the complete Stable Diffusion playlist](https://www.youtube.com/playlist?list=PLOW87P7MQGbE) · [Explore the project folder](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/tree/main/3%20Stable%20Diffusion%20from%20Scratch)


## PaliGemma from Scratch — Multimodal Vision-Language Model

A coding-focused PyTorch video series exploring PaliGemma and multimodal vision-language modelling, connecting image understanding with language capabilities.


<a href="https://www.youtube.com/playlist?list=PLKIurYhZ0z8o"><img src="https://img.youtube.com/vi/3tIQ9wJwy8Y/hqdefault.jpg" alt="Coding Multimodal Vision PaliGemma from Scratch video thumbnail" width="480"></a>


[Watch the complete PaliGemma playlist](https://www.youtube.com/playlist?list=PLKIurYhZ0z8o) · [Explore the PaliGemma project repository](https://github.com/AliHassanAbbas/PaliGemma-training-inference-pytorch)


## Swin Transformer from Scratch — Hierarchical Vision Transformer

A PyTorch architecture implementation based on the core ideas of Swin Transformer V1. The notebook builds a four-stage hierarchical vision model and demonstrates a forward pass on a dummy 224 × 224 RGB image.

**Implemented components and concepts**
- Patch partitioning and embedding with nn.Unfold and linear projection
- Hierarchical stages with progressive spatial downsampling and channel expansion
- Window-based multi-head self-attention
- Alternating regular-window and shifted-window attention blocks
- Cyclic shifts and attention masks to enable cross-window information flow
- Learned relative position bias
- Patch merging, residual connections, Pre-LayerNorm, and GELU feed-forward networks
- Configurable Tiny, Small, Base, and Large-style model factory functions

**Important scope note:** the inspected notebook implements the model architecture and a dummy-input forward pass. It does not contain a dataset/training/evaluation pipeline or trained weights. It follows Swin Transformer V1 ideas; the linked V2 paper is included as further reading, not as a separate V2 implementation.


[Read the detailed Swin Transformer README](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/blob/main/4%20Swin%20Transformer%20from%20Scratch/README.md) · [Open the notebook](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/blob/main/4%20Swin%20Transformer%20from%20Scratch/Coding_Swin_Transformer_from_scratch_in_pytorch.ipynb) · [Watch the 4-part video series](https://www.youtube.com/playlist?list=PLVg4rZCmnzpo)


## LLaMA 2 from Scratch — PyTorch Architecture & Inference

A notebook implementing core LLaMA 2 inference components: Transformer blocks, RMSNorm, Rotary Positional Embeddings (RoPE), multi-head attention with Grouped-Query Attention support, KV caching, and a SwiGLU-style feed-forward network. Includes SentencePiece tokenization, checkpoint loading, incremental generation, temperature sampling, top-p sampling, and greedy decoding.


**Scope:** This is an architecture and inference implementation that loads LLaMA 2 checkpoint weights; it is not a pretraining-from-random-initialization project.


[Read the detailed README](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/blob/main/5%20LLaMA%202%20from%20scratch/README.md) · [Open notebook](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/blob/main/5%20LLaMA%202%20from%20scratch/Coding_LLaMA_2_from_scratch.ipynb)


## U-Net from Scratch in PyTorch — Biomedical Image Segmentation

A custom encoder–decoder CNN for binary segmentation trained and evaluated on TIFF volume data. The notebook implements convolution–BatchNorm–ReLU blocks, max-pooling, a 1024-channel bottleneck, transposed-convolution upsampling, concatenative skip connections, and a 1×1 output projection. The pipeline includes TIFF frame splitting, custom Dataset/DataLoader, grayscale normalization, paired random flips, BCEWithLogitsLoss, Adam, 100-epoch training, validation, checkpoint resume/save, TensorBoard visualization, and held-out test predictions. The notebook reports a test average loss of approximately 0.2504 from its saved run (not a benchmark claim).


[Read the U-Net README](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/blob/main/9%20Unet%20from%20scratch%20in%20pytorch/README.md) · [Open the notebook](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/blob/main/9%20Unet%20from%20scratch%20in%20pytorch/UNET_from_scrtch_in_pytorch%20(2).ipynb)


## Normalization Methods in Deep Learning — PyTorch

A comparative notebook study of BatchNorm, LayerNorm, InstanceNorm, GroupNorm, and RMSNorm, including normalization axes, tensor-shape reasoning, learnable affine parameters, epsilon, custom PyTorch modules, and built-in implementations. Covers normalization choices in CNNs, image-generation models, and Transformer/LLM architectures.


[Read the normalization methods README](https://github.com/AliHassanAbbas/All-Normalizations-Explained-Batch-Layer-Instance-Group-RMS-PyTorch-/blob/main/README.md) · [Open the notebook](https://github.com/AliHassanAbbas/All-Normalizations-Explained-Batch-Layer-Instance-Group-RMS-PyTorch-/blob/main/All_Normalizations_Explained_Batch,_Layer,_Instance,_Group,_RMS_(+_PyTorch).ipynb)


## Technical Expertise Demonstrated

- PyTorch model development: custom layers and modules, tensor operations, broadcasting, reduction dimensions, and parameter initialization
- Deep learning architectures: CNNs, encoder–decoder networks, Transformers, language models, and generative models
- Normalization methods: BatchNorm, LayerNorm, InstanceNorm, GroupNorm, RMSNorm; understanding their axes, statistics, and use cases
- Training workflows: datasets and dataloaders, augmentation, loss functions, optimizers, validation, checkpoints, and TensorBoard
- Computer vision and NLP fundamentals, architecture implementation, and from-scratch educational experiments


## GPT-2 from Scratch — project scaffold

The [GPT-2 from Scratch folder](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/tree/main/8%20GPT%202%20from%20Scratch) now has an introductory README describing the intended learning scope. **The implementation notebook/source code has not yet been uploaded**, so implementation-specific architecture details and technical skills are not claimed for this folder yet. The related [large-language-model video series](https://www.youtube.com/playlist?list=PLEq4YdBohZDQ) is available on YouTube.
