# Swin Transformer from Scratch with PyTorch

A notebook-based implementation study of the **Swin Transformer (Shifted Window Transformer)** for hierarchical visual representation learning. The notebook builds the main architecture components in PyTorch and demonstrates a forward pass on a dummy image tensor.

- **Notebook:** [Coding_Swin_Transformer_from_scratch_in_pytorch.ipynb](./Coding_Swin_Transformer_from_scratch_in_pytorch.ipynb)
- **Video series:** [Swin Transformer — 4-part YouTube playlist](https://www.youtube.com/playlist?list=PLVg4rZCmnzpo)
- **Research paper:** [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (2021)](https://arxiv.org/abs/2103.14030)
- **Related paper:** [Swin Transformer V2: Scaling Up Capacity and Resolution (2021)](https://arxiv.org/abs/2111.09883)

## What the notebook implements

The notebook defines the architecture as a hierarchy of four stages:

1. **Patch partition / embedding:** uses `nn.Unfold` to extract non-overlapping image patches and a linear projection to map them to feature dimensions.
2. **Swin stages:** progressively increase channel dimensions while reducing spatial resolution.
3. **Swin blocks:** alternate regular-window and shifted-window attention blocks, each followed by a feed-forward network with residual connections and pre-layer normalization.
4. **Classification head:** globally averages the final feature map and applies LayerNorm and a linear classifier.

### Core building blocks

- **Window-based multi-head self-attention:** computes attention locally within non-overlapping windows rather than across every image token.
- **Shifted-window attention:** applies cyclic spatial shifts so information can flow across neighbouring windows in successive blocks.
- **Attention masks:** masks wrap-around connections introduced by cyclic shifts.
- **Relative position bias:** adds learned spatial-position information to attention scores.
- **Patch merging / downsampling:** extracts and projects patches to build a multi-scale feature hierarchy.
- **Feed-forward network:** two linear layers with GELU activation.
- **Residual connections and Pre-LayerNorm:** used around attention and feed-forward sublayers.

## Architecture configuration

The notebook includes factory functions for several model configurations:

| Variant helper | Stage depths | Attention heads |
|---|---|---|
| `swin_t` | (2, 2, 6, 2) | (3, 6, 12, 24) |
| `swin_s` | (2, 2, 18, 2) | (3, 6, 12, 24) |
| `swin_b` | (2, 2, 18, 2) | (4, 8, 16, 32) |
| `swin_l` | (2, 2, 18, 2) | (6, 12, 24, 48) |

The final example instantiates a model with a 96-dimensional initial hidden representation, stage depths (2, 2, 6, 2), window size 7, downscaling factors (4, 2, 2, 2), and a three-class output head. It runs a forward pass using a random tensor of shape `(1, 3, 224, 224)`.

## Technical concepts demonstrated

- PyTorch model construction with `nn.Module`, `nn.ModuleList`, `nn.Linear`, `nn.LayerNorm`, and `nn.Unfold`
- Tensor reshaping and dimension manipulation with `einops.rearrange`
- Query, key, and value projections for multi-head attention
- Scaled dot-product attention and softmax
- Local-window attention and shifted-window masking
- Cyclic shifts with `torch.roll`
- Relative position indexing and learned position bias
- Hierarchical feature maps and stage-wise channel scaling
- Residual learning, Pre-LayerNorm, and GELU feed-forward layers
- Configurable model depth, head counts, window size, and feature dimensions

## Run the notebook

Open the notebook in Jupyter or Google Colab. The imports used by the implementation include:

```bash
pip install torch numpy einops
```

The notebook's final cell performs a shape/forward-pass demonstration. The default example uses a 224 × 224 RGB input; the spatial dimensions must remain compatible with the configured downsampling stages and attention-window partitioning.

## Scope and implementation notes

This repository contains an **architecture implementation and forward-pass demonstration**, not a complete model-training experiment. The inspected notebook does not include a dataset pipeline, loss function, optimizer, training loop, validation metrics, checkpointing, or trained weights. The three-class head in the example is a configurable output layer, not evidence of a completed three-class training run.

The implementation follows the core Swin Transformer V1 ideas. Although the notebook links the Swin Transformer V2 paper for further reading, it does not implement the defining V2 changes as a separate model.

For a production or GPU training workflow, review device handling for the relative-position index tensor (it is created as a regular tensor rather than registered as a module buffer), validate input/window divisibility, and add a complete dataset, training, evaluation, and checkpoint pipeline.

## References

- Liu et al., [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030), ICCV 2021.
- Liu et al., [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883), CVPR 2022.

---

**Author:** [Ali Hassan Abbas](https://github.com/AliHassanAbbas) · [YouTube series](https://www.youtube.com/playlist?list=PLVg4rZCmnzpo)
