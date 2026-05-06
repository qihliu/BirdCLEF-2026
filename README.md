# BirdCLEF+ 2026

A PyTorch baseline for the [BirdCLEF+ 2026 Kaggle competition](https://www.kaggle.com/competitions/birdclef-2026) — acoustic species identification from passive acoustic monitoring recordings in the Pantanal wetlands, South America.

## Competition Overview

~1,000 acoustic recorders were deployed across the Pantanal (150,000+ km²), capturing audio from 234 wildlife species (birds, frogs, insects, reptiles, mammals). The task is to identify which species are present in each 5-second window of a soundscape recording.

**Evaluation metric:** Padded class-mean Average Precision (padded cmAP)  
**Submission deadline:** June 3, 2026

## Approach

```
Audio clip / soundscape window
        │
        ▼
  Mel Spectrogram  (128 mel bins × ~500 time frames per 5 s)
  → power-to-dB → normalise [0,1] → resize 256×256 → ImageNet normalise
        │
        ▼
  EfficientNet-B4 NoisyStudent  (tf_efficientnet_b4_ns, pretrained)
        │
        ▼
  Linear head  →  234-class logits
        │
        ▼
  BCEWithLogitsLoss  (multi-label, with pos_weight for class imbalance)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **`tf_efficientnet_b4_ns` backbone** | NoisyStudent semi-supervised pretraining on 300M images gives noise-robust features — the community's go-to backbone for BirdCLEF 2022–2024 |
| **Mel spectrogram → image** | Converts audio to a 2D image; unlocks large pretrained vision models |
| **`hop_length=320`** | Exactly 100 frames/sec → clean 500-frame window for 5 s |
| **ImageNet normalisation** | Aligns mel-spec pixel statistics with NoisyStudent pretraining |
| **Multi-label BCE loss** | Multiple species can overlap in a single 5-second window |
| **Soundscape training data** | 739 labeled windows from PAM recordings bridge the domain gap to the test set |
| **Domain-matched validation** | Soundscape val set (13 held-out files) is used for checkpoint selection; clean-clip val tracks learning curves across all 234 species |
| **WeightedRandomSampler + pos_weight** | Addresses 499× class imbalance (1–499 clips per species) |
| **SpecAugment + Mixup** | Regularises rare species; improves probability calibration |
| **5-fold ensemble** | Averages sigmoid outputs across folds at inference time |

## Repository Structure

```
├── config.py              # All hyperparameters (CFG class)
├── transforms.py          # AudioTransform, MelSpecTransform, SpecAugment, Mixup
├── dataset.py             # BirdTrainDataset, SoundscapeTrainDataset, SoundscapeInferenceDataset
├── model.py               # BirdModel (timm backbone + linear head)
├── train.py               # K-fold training script
├── inference.py           # Local inference → submission.csv
├── submission.ipynb       # Self-contained Kaggle submission notebook
├── kernel-metadata.json   # Kaggle CLI push config (kaggle kernels push .)
└── requirements.txt       # Python dependencies
```

## Setup

```bash
git clone https://github.com/qihliu/BirdCLEF-2026.git
cd BirdCLEF-2026
pip install -r requirements.txt

# Download competition data (requires Kaggle API key)
kaggle competitions download -c birdclef-2026
unzip birdclef-2026.zip
```

> Update `LOCAL_DATA_DIR` in `config.py` to point to your data directory.

## Training

```bash
# Quick start: fold 0 only (~1–2 h on a GPU)
python train.py

# All 5 folds for the full ensemble
python train.py --all-folds

# Quality filter: only clips rated ≥ 3 (faster, less noise)
python train.py --min-rating 3.0 --folds 0

# Custom settings
python train.py --epochs 20 --batch-size 16
```

Each epoch logs both `clip_cmAP` (all 234 species, stable) and `snd_cmAP*` (domain-matched to test, primary checkpoint criterion). Checkpoints are saved to `outputs/fold{N}_best.pth` when `snd_cmAP` improves.

## Local Inference

```bash
python inference.py
```

Loads all `outputs/fold*_best.pth` checkpoints, ensembles over `test_soundscapes/`, and writes `outputs/submission.csv`. Locally `test_soundscapes/` is empty — Kaggle populates it during the hidden scoring rerun.

## Kaggle Submission

**One-time setup:**
```bash
pip install kaggle
# Download kaggle.json from kaggle.com → Settings → API
mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

**Workflow:**
1. Train: `python train.py --all-folds`
2. Upload `outputs/fold*_best.pth` to a **private Kaggle Dataset** named `birdclef2026-checkpoints`
3. Push the notebook to Kaggle via CLI:
   ```bash
   kaggle kernels push .
   ```
4. On Kaggle, open the kernel → **Save & Run All** (interactive session; `test_soundscapes/` is empty, produces a placeholder output — this is normal)
5. Click **Submit** — Kaggle runs the notebook again on the hidden test soundscapes and scores it

> **Two-session system:** The interactive "Save & Run All" just verifies the notebook doesn't crash. The actual scoring uses a separate hidden rerun where Kaggle injects the real test files.

## Data

All data is available on the [competition data page](https://www.kaggle.com/competitions/birdclef-2026/data).

| File | Description |
|---|---|
| `train.csv` | 35,549 labeled short clips across 206 species |
| `train_audio/` | Audio clips organised by species code (OGG) |
| `train_soundscapes/` | 10,658 long soundscapes (~1–2 min, mostly unlabeled) |
| `train_soundscapes_labels.csv` | 739 labeled 5-second windows from 66 soundscapes |
| `taxonomy.csv` | 234 species with scientific names and taxonomy class |
| `sample_submission.csv` | Submission format template |
| `test_soundscapes/` | Hidden test set — populated by Kaggle at scoring time only |

## Spectrogram Parameters

| Parameter | Value |
|---|---|
| Sample rate | 32,000 Hz |
| n_mels | 128 |
| n_fft | 1,024 |
| hop_length | 320 → 100 frames/sec |
| fmin / fmax | 20 Hz / 16,000 Hz |
| Image size | 256 × 256 |

---

## How to Improve Results

The current baseline is a solid starting point. Below are concrete improvements roughly ordered by expected impact.

### 1. Better augmentation (easiest wins)

- **Background noise mixing:** Download [BirdCLEF background noise files](https://www.kaggle.com/datasets/wiradkp/esc50-for-birdclef) and mix them into training clips at random SNR (signal-to-noise ratio). This directly mimics the noisy PAM environment of the test set.
- **Pitch shift / time stretch:** `librosa.effects.pitch_shift` and `librosa.effects.time_stretch` teach the model that a bird call is the same species at slightly different speeds and pitches.
- **Label smoothing:** Replace hard 0/1 targets with 0.05/0.95 — prevents the model from becoming overconfident on noisy labels.

### 2. Overlapping inference windows

Currently inference uses non-overlapping 5-second windows. Using a **50% step** (sliding every 2.5 s) and averaging predictions for each second doubles the number of predictions and smooths out edge effects where a call straddles a window boundary.

### 3. Better backbone

| Model | Params | Notes |
|---|---|---|
| `tf_efficientnet_b4_ns` (current) | 19 M | Proven baseline |
| `efficientnetv2_s` | 22 M | Better architecture, similar speed |
| `convnext_small` | 50 M | Strong on spectrograms, top BirdCLEF 2024 solutions |
| `BEATs` (Microsoft) | 90 M | Audio-native transformer, pretrained on AudioSet; top BirdCLEF 2025 solutions |

### 4. Two-stage fine-tuning

Instead of training the whole model from epoch 1:
1. **Stage 1 (5 epochs):** Freeze the backbone, train only the classification head at high LR (1e-2). The backbone weights are preserved.
2. **Stage 2 (25 epochs):** Unfreeze everything, train at low LR (1e-4). The head is already sensible, so the backbone adapts without catastrophic forgetting.

### 5. Semi-supervised learning on unlabeled soundscapes

10,658 soundscapes exist without labels. After training a first model:
1. Run inference on all unlabeled soundscapes → pseudo-labels (predictions above a threshold, e.g. 0.7)
2. Add those windows to the training set with their pseudo-labels
3. Retrain

This is called **pseudo-labeling** and can give significant gains when labeled data is scarce.

### 6. Test-Time Augmentation (TTA)

At inference, run each window through the model multiple times with slight variations (horizontal flip of the spectrogram, small time shifts) and average the results. Easy to implement, free performance.

### 7. Sound Event Detection (SED) head

Rather than a single prediction per 5-second window, an SED head predicts frame-level presence (e.g., one score per 0.1 s). The clip-level prediction is the max (or mean) over frames. This gives finer-grained localisation and often improves cmAP because the model learns where in the window the call occurs.

---

## What to Learn

### Core prerequisites (if not already solid)
- **CNNs end-to-end:** Understand convolution, pooling, batch normalisation, skip connections (ResNet), and depthwise separable convolution (EfficientNet). *Resource: fast.ai Part 1.*
- **Transfer learning:** Why fine-tuning works, learning rate strategies (layer-wise LR decay, gradual unfreezing). *Resource: "How to fine-tune BERT" paper — same ideas apply to vision models.*
- **Loss functions for multi-label classification:** BCE vs. focal loss vs. ASL (Asymmetric Loss). *Resource: the ASL paper (Ben-Baruch et al., 2021).*

### Audio-specific knowledge
- **Digital signal processing basics:** Sampling theorem, DFT, windowing (Hann window). Understanding *why* n_fft and hop_length matter. *Resource: Julius O. Smith's online DSP book (free).*
- **Mel filterbank:** Why logarithmic frequency spacing matches human/animal hearing. *Resource: librosa's documentation has a good visual explanation.*
- **Constant-Q Transform (CQT):** An alternative to mel spectrograms that uses logarithmically-spaced frequency bins — often better for pitched sounds like bird songs. *Resource: librosa.cqt docs.*

### Competition-specific techniques
- **Sound Event Detection (SED):** Frame-level classification, attention-based pooling (GWRP — global weighted rank pooling). *Resource: DCASE challenge papers, especially 2020–2022.*
- **Domain adaptation:** Batch normalisation statistics differ between clean clips and soundscapes. Techniques: domain-adversarial training, test-time batch normalisation. *Resource: "Domain Adaptation for Audio" survey.*
- **Pseudo-labeling / knowledge distillation:** Using a teacher model to label unlabeled data for a student. *Resource: the NoisyStudent paper — you're already using its weights.*
