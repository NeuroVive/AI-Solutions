# NeuroVox — Parkinson's Disease Detection from Voice

> A deep learning system for non-invasive Parkinson's Disease detection from voice recordings. NeuroVox converts speech audio into Mel-Spectrograms and classifies them using a fine-tuned ResNet-18 backbone, achieving **97.80% validation accuracy** in just 6 epochs.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [API](#api)
- [Project Structure](#project-structure)
- [Documentation](#documentation)

---

## Overview

Parkinson's Disease (PD) causes characteristic changes in vocal production — reduced loudness, monotone speech, and increased breathiness — which can be captured in audio recordings. NeuroVox automates this analysis by:

1. Converting raw `.wav` recordings into **Mel-Spectrogram** representations
2. Applying data augmentation (time-stretch, pitch-shift, noise injection) to enrich training data
3. Classifying spectrograms with a **ResNet-18 transfer learning** model fine-tuned on single-channel audio input
4. Serving predictions through a **FastAPI** REST endpoint

Labels: `HC` (Healthy Control) · `PD` (Parkinson's Disease)

---

## Features

| Feature                    | Description                                                     |
| -------------------------- | --------------------------------------------------------------- |
| Audio preprocessing        | Chunking, padding, silence filtering, energy validation         |
| Mel-Spectrogram extraction | 40-band log-scaled spectrograms via librosa                     |
| Data augmentation          | Time-stretch, pitch-shift, Gaussian noise per chunk             |
| Transfer learning          | ResNet-18 (ImageNet) adapted for 1-channel spectrogram input    |
| ONNX export                | Production-ready inference without PyTorch dependency           |
| REST API                   | FastAPI endpoint with async audio upload and background cleanup |
| Training visualization     | Loss, Accuracy, F1 curves + Confusion Matrix                    |

---

## Architecture

```
Raw WAV File
     │
  Validation (duration, energy, integrity)
     │
  Audio Chunking (6s windows, 10% overlap)
     │
  Augmentation (time-stretch / pitch-shift / noise)
     │
  Mel-Spectrogram (40 mels, n_fft=1024)
     │
  ┌──────────────────────────────┐
  │  ResNet-18 Backbone          │
  │  (conv1 → 1-channel input)   │
  │  → 512-d embedding           │
  └──────────┬───────────────────┘
             │
       Custom Head
    Linear(512→256) → BN → SiLU → Dropout → Linear(256→1)
             │
        Raw Logit → Sigmoid → Label (HC / PD)
```

---

## Results

| Metric                    | Value                |
| ------------------------- | -------------------- |
| Best Validation Accuracy  | **97.80%** (Epoch 6) |
| Best Validation F1        | **97.37%** (Epoch 6) |
| Training Accuracy (final) | **100.00%**          |
| Total Epochs              | 10                   |
| Valid Audio Files         | 1,274                |

See [`docs/results.md`](docs/results.md) for the full epoch-by-epoch breakdown.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/A-Ahmed-I/AI-Solutions.git
cd NeuroVox

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU users:** Install the CUDA build of PyTorch first:
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Usage

### Train the model

```bash
python -m src.main
```

### Start the API server

```bash
python -m src.api.main
# API available at http://localhost:8000
# Docs at      http://localhost:8000/docs
```

### Run inference directly

```python
from src.inference.predictor import NeuroVoxPredictor

predictor = NeuroVoxPredictor("checkpoint/best_model.onnx")
label, probability = predictor.predict("recording.wav")

print(f"Prediction : {label}")        # 'HC' or 'PD'
print(f"Probability: {probability}")  # e.g. 0.8921
```

---

## API

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -F "audio=@voice_sample.wav"
```

**Response:**

```json
{
  "label": "HC",
  "probability": 0.8921
}
```

See [`docs/api.md`](docs/api.md) for full reference.

---

## Project Structure

```
NeuroVox/
├── README.md                     ← Project overview (This file)
├── requirements.txt              ← Python dependencies
│
├── docs/
│   ├── overview.md               ← Architecture & design decisions
│   ├── data_pipeline.md          ← Preprocessing & augmentation
│   ├── model.md                  ← Model architectures
│   ├── training.md               ← Training loop & configuration
│   ├── api.md                    ← REST API reference
│   ├── inference.md              ← ONNX inference guide
│   └── results.md                ← Metrics & epoch logs
│
├── src/
│   ├── pipeline/
│   │   ├── main.py               ← Entry point for pipeline execution
│   │   └── pipeline.py           ← End-to-end training pipeline
│   │
│   ├── inference/
│   │   └── predictor.py          ← ONNX inference class
│   │
│   ├── plots/
│   │   └── plot.py               ← visualizations
│   │
│   ├── augmentation/
│   │   └── augmented.py          ← Data augmentation utilities
│   │
│   ├── data/
│   │   ├── data_loader.py        ← Dataset loading and metadata handling
│   │   ├── metadata.py           ← Dataset metadata handling
│   │   └── custom_data.py        ← Custom dataset loader
│   │
│   ├── feature_extraction/
│   │   └── handcrafted.py        ← Handcrafted feature extraction (HOG, LBP, etc.)
│   │
│   ├── processing/
│   │   └── processing.py         ← Data preprocessing pipeline
│   │
│   ├── training/
│   │   └── train.py              ← Model training loop
│   │
│   ├── model/
│   │   ├── neurovox_cnn.py       ← CNN-based model architecture
│   │   ├── neurovox_rn.py        ← Residual Network (ResNet-based) architecture
│   │   └── neurovox_tl.py        ← Transfer learning model architecture
│   │
│   ├── utils/
│   │   └── helper.py             ← Utility/helper functions
│   │
│   ├── api/
│   │   ├── endpoint.py           ← /predict routes implementation
│   │   └── main.py               ← FastAPI app entry point
│   │
│   ├── constant/
│   │   └── constant.py           ← Hyperparameters, paths, configs
│   │
│   └── __init__.py               ← Package initializer
```

---

## Documentation

| Document                                         | Description                           |
| ------------------------------------------------ | ------------------------------------- |
| [`docs/overview.md`](docs/overview.md)           | System architecture and motivation    |
| [`docs/data_pipeline.md`](docs/data_pipeline.md) | Audio preprocessing and augmentation  |
| [`docs/model.md`](docs/model.md)                 | Model architectures (CNN, ResNet, TL) |
| [`docs/training.md`](docs/training.md)           | Training configuration and scheduler  |
| [`docs/api.md`](docs/api.md)                     | REST API reference                    |
| [`docs/inference.md`](docs/inference.md)         | ONNX inference guide                  |
| [`docs/results.md`](docs/results.md)             | Performance metrics and training logs |

---

## License

This project is licensed under the MIT License.
