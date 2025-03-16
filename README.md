# PixelTapestry

PixelTapestry is a Python library for advanced image segmentation and embedding using neural networks based on the MONAI and PyTorch frameworks.

## Overview

PixelTapestry specializes in binary pixel embedding using modified Swin UNETR architecture. The library provides tools for creating and manipulating pixel-level representations in medical and scientific imaging.

## Current Features

- **BinaryEmbeddingSwinUNETR**: A modified version of the Swin UNETR architecture that returns binary embeddings
  - Training mode that produces differentiable binarized embeddings using sigmoid
  - Inference mode that produces hard binary embeddings
  - Seamless integration with MONAI's existing components

## Installation

```bash
pip install pixeltapestry
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/kephale/pixeltapestry.git
```

## Requirements

- Python ≥ 3.9
- PyTorch
- MONAI
- NumPy

## Usage

```python
import torch
from pixeltapestry.models import BinaryEmbeddingSwinUNETR

# Initialize the model
model = BinaryEmbeddingSwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=2,
    feature_size=48
)

# For inference
x = torch.randn(1, 1, 96, 96, 96)
binary_embedding = model(x)  # Returns a binary embedding

# For training
model.train()
x = torch.randn(1, 1, 96, 96, 96)
sigmoid_embedding, logits = model(x)  # Returns sigmoid embedding and segmentation logits
```

## Future Development Ideas

### Short-term Goals

1. **Additional Binary Embedding Architectures**
   - Add UNet and SegResNet variants with binary embedding outputs
   - Implement attention-based mechanisms for improved embedding quality

2. **Embedding Analysis Tools**
   - Functions for visualizing and analyzing the learned binary embeddings
   - Metrics for evaluating the quality of embeddings

3. **Pre-trained Models**
   - Provide pre-trained models for common imaging domains
   - Add support for transfer learning from pre-trained models

### Medium-term Goals

4. **Multi-modal Fusion**
   - Support for combining information from multiple imaging modalities
   - Cross-modal attention mechanisms for integrated analysis

5. **Temporal Embedding Tracking**
   - Tools for tracking embeddings across time series data
   - Temporal consistency constraints for dynamic imaging

6. **Hierarchical Embedding**
   - Implementation of hierarchical binary embeddings for multi-scale analysis
   - Coarse-to-fine segmentation approaches

### Long-term Vision

7. **Self-supervised Learning**
   - Framework for self-supervised learning of binary embeddings
   - Contrastive learning methods specialized for scientific imaging

8. **Integration with Other Tools**
   - Connect with visualization frameworks like napari
   - Provide formats compatible with standard analysis pipelines

9. **Domain-specific Extensions**
   - Specialized modules for microscopy, radiology, and other domains
   - Task-specific loss functions and architectures

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use PixelTapestry in your research, please cite:

```
@software{harrington2025pixeltapestry,
  author = {Harrington, Kyle},
  title = {PixelTapestry: Binary Pixel Embedding for Scientific Imaging},
  url = {https://github.com/kephale/pixeltapestry},
  year = {2025}
}
```