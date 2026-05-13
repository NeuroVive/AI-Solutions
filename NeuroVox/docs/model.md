# Model Architectures

NeuroVox provides three CNN architectures for spectrogram-based audio classification. All models share the same input format and output a single raw logit (apply `sigmoid` for probability).

**Input:** `(B, 1, 40, T)` — batch of single-channel Mel-Spectrograms  
**Output:** `(B, 1)` — raw logit per sample

---

## Primary Model — `NeuroVoxTL` (Transfer Learning)

**File:** `src/models/neurovox_tl.py`

This is the model used in production. It adapts a pretrained **ResNet-18** (trained on ImageNet) for single-channel spectrogram classification.

### Architecture

```
Input (B, 1, 40, T)
        │
Conv2d(1→64, kernel=(3,7), stride=(1,2)) ← replaces original first conv
        │
ResNet-18 Backbone (layers 1–4)
        │
AdaptiveAvgPool2d → 512-d embedding
        │
  ┌─────────────────────────────┐
  │  Custom Classifier Head     │
  │  Linear(512 → 256)          │
  │  BatchNorm1d(256)           │
  │  SiLU()                     │
  │  Dropout(0.5)               │
  │  Linear(256 → 1)            │
  └─────────────────────────────┘
        │
    Raw Logit (B, 1)
```

### Key Modifications from Standard ResNet-18

| Component         | Original ResNet-18             | NeuroVoxTL                         |
| ----------------- | ------------------------------ | ---------------------------------- |
| First convolution | `Conv2d(3, 64, 7×7, stride=2)` | `Conv2d(1, 64, 3×7, stride=(1,2))` |
| Final FC layer    | `Linear(512, 1000)`            | Replaced with `nn.Identity()`      |
| Classifier head   | Softmax (1000 classes)         | Custom binary head with SiLU       |

The asymmetric kernel `(3, 7)` and `stride=(1,2)` for the first convolution preserves more temporal resolution in the frequency axis — important for capturing the fine-grained spectral structure of voice pathology.

### Why ResNet-18?

- **Pretrained representations:** ImageNet features (edges, textures, patterns) transfer effectively to spectrograms, which are structured 2D images.
- **Depth vs. dataset size tradeoff:** With ~1,274 audio files (expanded by augmentation), ResNet-18 strikes the right balance — deep enough to learn discriminative features, light enough to avoid overfitting.
- **Proven stability:** Residual connections prevent vanishing gradients, enabling reliable convergence even with a small dataset.

---

## Alternative Model — `NeuroVoxCNN` (Custom CNN)

**File:** `src/models/neurovox_cnn.py`

A simpler 3-block CNN designed for quick experimentation or resource-constrained environments.

### Architecture

```
Input (B, 1, H, W)
        │
Block 1: Conv2d(1 → C,    k=3) → BN → ReLU → MaxPool(3)
Block 2: Conv2d(C → 2C,   k=2) → BN → ReLU → MaxPool(2)
Block 3: Conv2d(2C → 4C,  k=3) → BN → ReLU → MaxPool(3)
        │
AdaptiveAvgPool2d(1, 1)
        │
Flatten → Linear(4C → 2C) → BN → ReLU → Dropout(0.5)
       → Linear(2C → C)  → BN → ReLU → Dropout(0.5)
       → Linear(C → out_ch)
        │
    Raw Logit
```

### Constructor Parameters

| Parameter      | Description                         |
| -------------- | ----------------------------------- |
| `input_ch`     | Input channels (1 for spectrograms) |
| `hidden_ch`    | Base channel width (e.g., 32 or 64) |
| `out_ch`       | Output classes (1 for binary)       |
| `dropout_rate` | Dropout probability (default: 0.5)  |

---

## Alternative Model — `NeuroVoxRN` (Custom ResNet)

**File:** `src/models/neurovox_rn.py`

A lightweight custom residual network that sits between `NeuroVoxCNN` and the full transfer learning model in terms of complexity.

### Residual Block

```
Input (B, C_in, H, W)
     │
Conv2d(3×3) → BN → ReLU
     │
Conv2d(3×3) → BN
     │
     + ── Shortcut (Identity or 1×1 projection if shapes differ)
     │
   ReLU
```

### Full Architecture

```
Conv2d(1→C, 3×3) → BN → ReLU
        │
ResidualBlock(C  → C)   → MaxPool(2)
ResidualBlock(C  → 2C)  → MaxPool(2)
ResidualBlock(2C → 4C)
        │
AdaptiveAvgPool2d(1, 1)
        │
Flatten → Dropout → Linear(4C → 2C) → BN → ReLU → Dropout → Linear(2C → 1)
```

---

## Model Comparison

| Model         | Pretrained  | Parameters | Complexity | Best For                        |
| ------------- | ----------- | ---------- | ---------- | ------------------------------- |
| `NeuroVoxTL`  | ✅ ImageNet | ~11M       | High       | **Production** — best accuracy  |
| `NeuroVoxRN`  | ❌          | ~0.5M      | Medium     | Ablation / moderate resources   |
| `NeuroVoxCNN` | ❌          | ~0.2M      | Low        | Fast prototyping / edge devices |

---

## ONNX Export

After training, the model is exported to ONNX for runtime-agnostic deployment:

```python
torch.onnx.export(
    model,
    dummy_input,            # shape: (1, 1, 40, 517)
    onnx_path,
    opset_version=11,
    input_names=["input"],
    output_names=["outputs"],
    dynamic_axes={
        "input":   {0: "batch_size"},
        "outputs": {0: "batch_size"},
    },
)
```

Dynamic batch size allows the ONNX model to handle single-sample and batched inference without re-export.
