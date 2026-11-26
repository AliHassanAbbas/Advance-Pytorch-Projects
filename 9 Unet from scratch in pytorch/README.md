# U-Net from Scratch in PyTorch — Biomedical Image Segmentation

A PyTorch implementation of the U-Net encoder–decoder architecture for image segmentation, following the original U-Net research paper. The notebook trains and evaluates a model on the ISBI 2012-style volume data provided as TIFF files.

## Notebook

[Open the U-Net notebook](https://github.com/AliHassanAbbas/Advance-Pytorch-Projects/blob/main/9%20Unet%20from%20scratch%20in%20pytorch/UNET_from_scrtch_in_pytorch%20(2).ipynb)

## Architecture

The network is implemented as a custom PyTorch module and includes:

- **Encoder / contracting path:** convolution blocks with Conv2D, BatchNorm2D and ReLU; max-pooling progressively reduces spatial resolution and expands feature channels (64 → 128 → 256 → 512 → 1024).
- **Bottleneck:** deepest representation at 1024 channels.
- **Decoder / expanding path:** transposed convolutions for upsampling, concatenation with matching encoder features (skip connections), and convolution blocks to recover spatial detail.
- **Output projection:** 1×1 convolution producing a single-channel segmentation logit map.

The implementation uses concatenative skip connections to combine spatial details from the encoder with higher-level features from the decoder.

## Data pipeline

The notebook works with TIFF stacks and uses the following files:

- train-volume.tif
- train-labels.tif
- test-volume.tif

It iterates over the TIFF frames, converts image/label pairs to NumPy arrays, shuffles the pairs and partitions them into training, validation and test subsets (24 / 3 / remaining frames in the notebook). It writes individual PNG frames and masks into organized directories.

A custom PyTorch Dataset reads grayscale image and label pairs, scales pixel values, and returns tensors in channel-first format. The transform pipeline includes grayscale normalization (mean 0.5, standard deviation 0.5), conversion to tensors, and random horizontal/vertical flips for training images and corresponding masks.

## Training and evaluation

- Loss: BCEWithLogitsLoss
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 2
- Training duration: 100 epochs
- Device selection: CUDA when available, otherwise CPU
- Validation: runs without gradient computation and records average loss
- Testing: evaluates a held-out split and saves input, ground-truth mask and thresholded prediction images
- Checkpoints: saves model and optimizer state per epoch and supports resuming
- Visualization: TensorBoard image and scalar logging for training and validation

The notebook includes a reported test average loss of approximately **0.2504** in its saved output. This is the loss value from that notebook run, not a benchmark claim.

## Skills demonstrated

- Implementing encoder–decoder CNNs in PyTorch
- Semantic / binary image segmentation
- U-Net skip connections and feature-map concatenation
- Transposed convolution (learned upsampling)
- Custom Dataset and DataLoader workflows
- TIFF stack handling and image/mask preprocessing with NumPy, OpenCV and PIL
- Paired image/mask augmentation
- Training loops, gradient backpropagation and validation
- BCE-with-logits loss and Adam optimization
- Checkpoint saving and resuming
- TensorBoard monitoring and segmentation result visualization
- GPU/CPU device-aware execution

## Setup and dataset

Install the notebook dependencies in your environment (PyTorch, torchvision, NumPy, OpenCV, Pillow, Matplotlib and TensorBoard). Download the three TIFF data files listed above and place them in an ISBI2012 folder as instructed in the notebook before running the cells.

This project is an educational implementation. Dataset splitting, preprocessing, training settings and saved notebook results reflect this notebook; they should not be interpreted as a medical-grade validation or a clinical performance claim.

## Reference

Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.*  
[Paper (arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

---

**Author:** Ali Hassan Abbas  
[GitHub profile](https://github.com/AliHassanAbbas)
