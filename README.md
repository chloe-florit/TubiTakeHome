# Human vs AI-Generated Images
## Convolutional Neural Network (CNN)

**Author:** Chloe Florit  

---

## Overview

This project builds a Convolutional Neural Network (CNN) in PyTorch to classify images as:

- **0 → Human-generated (Shutterstock)**  
- **1 → AI-generated**

The goal is to detect authenticity in digital images using deep learning and spatial feature extraction.

---

## Dataset

- 79,950 total images  
- Perfectly balanced (50% human, 50% AI)  
- 80% training / 20% validation split

Because the dataset is balanced, accuracy is used as the primary evaluation metric.

---

## Preprocessing

Images are:

- Resized to **224 × 224**
- Converted to tensors
- Normalized to **[-1, 1]** across RGB channels

Each input has shape:

```
[3, 224, 224]
```

---

## Model Architecture

### Convolutional Layers

- Conv1: 3 → 32 (3×3), ReLU, 2×2 MaxPool  
- Conv2: 32 → 64 (3×3), ReLU, 2×2 MaxPool  
- Conv3: 64 → 128 (3×3), ReLU, 2×2 MaxPool

After pooling:

```
224 → 112 → 56 → 28
```

Flattened feature size:

```
128 × 28 × 28 = 100,352
```

### Fully Connected Layers

- FC1: 100,352 → 512 (ReLU)  
- FC2: 512 → 2 (logits)

---

## Training Setup

- Loss: `CrossEntropyLoss`
- Optimizer: Adam (lr = 0.001)
- Batch size: 32
- Epochs: 2
- GPU used if available

---

## Results

| Epoch | Training Loss | Validation Accuracy |
|--------|---------------|--------------------|
| 1 | 0.2793 | 94.05% |
| 2 | 0.1292 | 97.30% |

The model converged quickly and achieved **97.30% validation accuracy** after two epochs.

---

## Contribution

Chloe implemented the CNN architecture, custom dataset class, training loop, and evaluation pipeline in PyTorch.

