import random
from typing import Optional, Tuple

import librosa
import numpy as np
from PIL import Image

from config import CFG

# ImageNet mean/std for normalisation (aligns with NoisyStudent pretraining)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class AudioTransform:
    """Crop or pad a mono waveform to exactly N_SAMPLES samples."""

    def __init__(self, cfg: CFG, is_train: bool = True):
        self.n_samples = cfg.N_SAMPLES
        self.is_train  = is_train

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        # Ensure 1-D float32
        waveform = np.squeeze(waveform).astype(np.float32)
        if waveform.ndim != 1 or len(waveform) == 0:
            return np.zeros(self.n_samples, dtype=np.float32)

        # Tile-repeat if shorter than target
        if len(waveform) < self.n_samples:
            repeats = (self.n_samples // len(waveform)) + 1
            waveform = np.tile(waveform, repeats)

        # Crop
        if self.is_train:
            max_start = len(waveform) - self.n_samples
            start = random.randint(0, max_start) if max_start > 0 else 0
        else:
            start = (len(waveform) - self.n_samples) // 2

        waveform = waveform[start : start + self.n_samples]

        # Amplitude normalisation (avoids NaN on silence)
        max_abs = np.max(np.abs(waveform))
        waveform = waveform / (max_abs + 1e-6)
        return waveform


class MelSpecTransform:
    """
    Convert a mono waveform (N_SAMPLES,) → normalised 3-channel image (3, IMG_SIZE, IMG_SIZE).

    Steps:
        1. Mel spectrogram (librosa)
        2. Power-to-dB (ref=1.0, top_db clamped)
        3. Normalise to [0, 1]
        4. Resize to IMG_SIZE × IMG_SIZE
        5. Stack to 3 channels
        6. ImageNet normalise (aligns with NoisyStudent pretrain)
    """

    def __init__(self, cfg: CFG):
        self.sr         = cfg.SAMPLE_RATE
        self.n_mels     = cfg.N_MELS
        self.n_fft      = cfg.N_FFT
        self.hop_length = cfg.HOP_LENGTH
        self.fmin       = cfg.FMIN
        self.fmax       = cfg.FMAX
        self.top_db     = cfg.TOP_DB
        self.img_size   = cfg.IMG_SIZE

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=1.0, top_db=self.top_db)

        # Normalise to [0, 1]  (-top_db..0 dB → 0..1)
        mel_norm = (mel_db + self.top_db) / self.top_db
        mel_norm = mel_norm.clip(0.0, 1.0).astype(np.float32)

        # Resize to (IMG_SIZE, IMG_SIZE) via PIL
        img_uint8 = (mel_norm * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8).resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        img = np.array(img_pil, dtype=np.float32) / 255.0

        # 3-channel stack then ImageNet normalise
        img = np.stack([img, img, img], axis=0)           # (3, H, W)
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        return img


class SpecAugment:
    """Frequency and time masking (SpecAugment) applied to a (3, H, W) image."""

    def __init__(self, cfg: CFG):
        self.freq_mask_max  = cfg.FREQ_MASK_MAX
        self.time_mask_max  = cfg.TIME_MASK_MAX
        self.num_freq_masks = cfg.NUM_FREQ_MASKS
        self.num_time_masks = cfg.NUM_TIME_MASKS

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _, H, W = img.shape
        img = img.copy()

        for _ in range(self.num_freq_masks):
            f = random.randint(0, min(self.freq_mask_max, H - 1))
            if f > 0:
                f0 = random.randint(0, H - f)
                img[:, f0 : f0 + f, :] = 0.0

        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_max, W - 1))
            if t > 0:
                t0 = random.randint(0, W - t)
                img[:, :, t0 : t0 + t] = 0.0

        return img


def mixup_data(
    x: "torch.Tensor",
    y: "torch.Tensor",
    alpha: float = 0.5,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", float]:
    """Batch-level mixup. Returns (mixed_x, y_a, y_b, lambda)."""
    import torch
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[perm]
    return mixed_x, y, y[perm], lam


def mixup_criterion(
    criterion,
    pred: "torch.Tensor",
    y_a: "torch.Tensor",
    y_b: "torch.Tensor",
    lam: float,
) -> "torch.Tensor":
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
