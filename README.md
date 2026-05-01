# BirdCLEF+ 2026

A PyTorch solution for the [BirdCLEF+ 2026 Kaggle competition](https://www.kaggle.com/competitions/birdclef-2026) — acoustic species identification from passive monitoring recordings in the Pantanal wetlands, South America.

## Competition Overview

~1,000 acoustic recorders were deployed across the Pantanal (150,000+ km²), capturing audio from 234 wildlife species (birds, frogs, insects, reptiles, mammals). The task is to classify which species are present in each 5-second window of a soundscape recording.

**Evaluation metric:** Padded class-mean Average Precision (padded cmAP)  
**Submission deadline:** June 3, 2026

## Approach

```
Audio clip / soundscape window
        │
        ▼
  Mel Spectrogram (128 × 500, librosa)
  + dB conversion + [0,1] normalisation
  + resize to 256 × 256
  + ImageNet normalisation
        │
        ▼
  EfficientNet-B4 NoisyStudent (tf_efficientnet_b4_ns)
  pretrained on ImageNet via NoisyStudent semi-supervised training
        │
        ▼
  Linear head → 234-class logits
        │
        ▼
  BCEWithLogitsLoss (multi-label)
```

### Key design choices

| Choice | Rationale |
|---|---|
| **`tf_efficientnet_b4_ns` backbone** | NoisyStudent pretraining (semi-supervised on 300M images) produces noise-robust features — consistently top-ranked in BirdCLEF 2022–2024 |
| **Mel spectrogram as image** | Converts the problem to image classification; leverages large pretrained vision models |
| **`hop_length=320`** | Exactly 100 frames/second → 5 s = 500 frames (clean, round number) |
| **ImageNet normalisation** | Aligns input statistics with NoisyStudent pretraining for faster convergence |
| **Multi-label BCE loss** | Multiple species can overlap within a 5-second window |
| **Soundscape training data** | Labeled windows from `train_soundscapes/` bridge the domain gap between clean XenoCanto recordings and PAM soundscapes |
| **SpecAugment + Mixup** | Regularisation for rare/imbalanced species |
| **5-fold ensemble** | Averages sigmoid probabilities across folds at inference time |

## Repository Structure

```
├── config.py          # All hyperparameters (CFG class)
├── transforms.py      # AudioTransform, MelSpecTransform, SpecAugment, Mixup
├── dataset.py         # BirdTrainDataset, SoundscapeTrainDataset, SoundscapeInferenceDataset
├── model.py           # BirdModel (timm backbone + linear head)
├── train.py           # Training script with k-fold cross-validation
├── inference.py       # Local inference → submission.csv
├── submission.ipynb   # Self-contained Kaggle submission notebook
└── requirements.txt   # Python dependencies
```

## Setup

```bash
# Clone the repo
git clone https://github.com/Hoshinanoriko/BirdCLEF-2026.git
cd BirdCLEF-2026

# Install dependencies
pip install -r requirements.txt

# Download competition data (requires Kaggle API key)
kaggle competitions download -c birdclef-2026
unzip birdclef-2026.zip
```

> **Note:** Update `LOCAL_DATA_DIR` in `config.py` to point to your data directory.

## Training

```bash
# Quick start: train fold 0 only (~1–2 h on a GPU)
python train.py

# Train all 5 folds for the full ensemble
python train.py --all-folds

# Use only high-quality clips (rating ≥ 3) for faster / cleaner training
python train.py --min-rating 3.0 --folds 0

# Custom epochs / batch size
python train.py --epochs 20 --batch-size 16
```

Checkpoints are saved to `outputs/fold{N}_best.pth` based on best validation cmAP.

## Local Inference

```bash
python inference.py
```

This loads all `outputs/fold*_best.pth` checkpoints, runs an ensemble over `test_soundscapes/`, and writes `outputs/submission.csv`.

> Locally, `test_soundscapes/` is empty — the hidden test set is only populated when Kaggle runs your notebook during scoring.

## Kaggle Submission

1. Train locally (or on Colab/Kaggle GPU): `python train.py --all-folds`
2. Upload the resulting `outputs/fold*_best.pth` files to a **private Kaggle Dataset** named `birdclef2026-checkpoints`
3. In `submission.ipynb`, verify `CKPT_DIR` points to that dataset
4. Add both the competition data and your checkpoint dataset to the notebook
5. Submit the notebook — Kaggle runs it against the hidden test soundscapes

## Data

All data is available from the [competition page](https://www.kaggle.com/competitions/birdclef-2026/data) and is not included in this repository.

| File | Description |
|---|---|
| `train.csv` | 35,549 labeled short clips (206 species) |
| `train_audio/` | Audio clips organised by species code |
| `train_soundscapes/` | 10,658 long soundscapes (mostly unlabeled) |
| `train_soundscapes_labels.csv` | 739 labeled 5-second windows from 66 soundscapes |
| `taxonomy.csv` | 234 species with scientific names and class |
| `sample_submission.csv` | Submission format template |
| `test_soundscapes/` | Hidden test set (populated by Kaggle at scoring time) |

## Spectrogram Parameters

| Parameter | Value |
|---|---|
| Sample rate | 32,000 Hz |
| n_mels | 128 |
| n_fft | 1,024 |
| hop_length | 320 (100 frames/sec) |
| fmin / fmax | 20 Hz / 16,000 Hz |
| Image size | 256 × 256 |
