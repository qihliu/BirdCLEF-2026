import ast
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from config import CFG
from transforms import AudioTransform, MelSpecTransform, SpecAugment


# ─────────────────────────────────────────────────────────────────────────────
# Species utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_species_info(taxonomy_csv_path: str) -> Dict:
    """
    Parse taxonomy.csv and return species lookup structures.

    Returns:
        {
            'species_list': List[str],        # 234 species in taxonomy order
            'label_to_idx': Dict[str, int],   # code → 0..233
            'idx_to_label': Dict[int, str],
        }
    """
    tax = pd.read_csv(taxonomy_csv_path)
    species_list = [str(x) for x in tax["primary_label"].tolist()]
    label_to_idx = {sp: i for i, sp in enumerate(species_list)}
    idx_to_label = {i: sp for sp, i in label_to_idx.items()}
    return {
        "species_list": species_list,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
    }


def build_multilabel_vector(
    primary: str,
    secondary: List[str],
    label_to_idx: Dict[str, int],
    n_classes: int,
    use_secondary: bool = True,
) -> np.ndarray:
    vec = np.zeros(n_classes, dtype=np.float32)
    pk = str(primary)
    if pk in label_to_idx:
        vec[label_to_idx[pk]] = 1.0
    if use_secondary:
        for s in secondary:
            sk = str(s)
            if sk in label_to_idx:
                vec[label_to_idx[sk]] = 1.0
    return vec


def _parse_secondary(raw: str) -> List[str]:
    """Parse secondary_labels stored as a Python-literal string like \"['abc', 'def']\"."""
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def _hms_to_sec(hms: str) -> int:
    """Convert 'HH:MM:SS' to integer seconds."""
    h, m, s = hms.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


# ─────────────────────────────────────────────────────────────────────────────
# Training dataset: short clips from train_audio/
# ─────────────────────────────────────────────────────────────────────────────

class BirdTrainDataset(Dataset):
    """
    Loads labeled short audio clips from train_audio/.

    Each item is a (image_tensor [3, IMG_SIZE, IMG_SIZE], label_vector [N_CLASSES]).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_base_dir: str,
        species_info: Dict,
        audio_transform: AudioTransform,
        mel_transform: MelSpecTransform,
        spec_augment: Optional[SpecAugment],
        cfg: CFG,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_base_dir = audio_base_dir
        self.label_to_idx = species_info["label_to_idx"]
        self.n_classes = len(self.label_to_idx)
        self.audio_tf = audio_transform
        self.mel_tf = mel_transform
        self.spec_aug = spec_augment
        self.use_secondary = cfg.USE_SECONDARY_LABELS

        # Precompute secondary label lists (avoid ast.literal_eval in hot path)
        self._secondary = [
            _parse_secondary(row["secondary_labels"])
            for _, row in self.df.iterrows()
        ]
        # Precompute label vectors
        self._labels = np.stack(
            [
                build_multilabel_vector(
                    str(row["primary_label"]),
                    self._secondary[i],
                    self.label_to_idx,
                    self.n_classes,
                    self.use_secondary,
                )
                for i, (_, row) in enumerate(self.df.iterrows())
            ],
            axis=0,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = os.path.join(self.audio_base_dir, str(row["filename"]))

        try:
            waveform, _ = librosa.load(path, sr=None, mono=True)
            if len(waveform) == 0:
                raise ValueError("empty audio")
        except Exception:
            waveform = np.zeros(self.audio_tf.n_samples, dtype=np.float32)

        waveform = self.audio_tf(waveform)
        img = self.mel_tf(waveform)
        if self.spec_aug is not None:
            img = self.spec_aug(img)

        label = self._labels[idx]
        return torch.from_numpy(img), torch.from_numpy(label)


# ─────────────────────────────────────────────────────────────────────────────
# Training dataset: labeled 5-second windows from train_soundscapes/
# ─────────────────────────────────────────────────────────────────────────────

class _LRUCache:
    """Simple LRU dict with a max-size cap (used for soundscape waveform cache)."""

    def __init__(self, maxsize: int = 10):
        self._cache: OrderedDict = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)


class SoundscapeTrainDataset(Dataset):
    """
    Labeled 5-second windows extracted from train_soundscapes/ soundscapes.

    Uses a per-worker LRU cache to avoid reloading the same soundscape file
    multiple times within one epoch.
    """

    def __init__(
        self,
        labels_csv_path: str,
        soundscapes_dir: str,
        species_info: Dict,
        audio_transform: AudioTransform,
        mel_transform: MelSpecTransform,
        spec_augment: Optional[SpecAugment],
        cfg: CFG,
    ):
        df = pd.read_csv(labels_csv_path)
        # Deduplicate: the CSV contains duplicate rows
        df = df.drop_duplicates(subset=["filename", "start", "end"]).reset_index(drop=True)

        self.df = df
        self.soundscapes_dir = soundscapes_dir
        self.label_to_idx = species_info["label_to_idx"]
        self.n_classes = len(self.label_to_idx)
        self.audio_tf = audio_transform
        self.mel_tf = mel_transform
        self.spec_aug = spec_augment
        self.sr = cfg.SAMPLE_RATE
        self.n_samples = cfg.N_SAMPLES
        self.use_secondary = cfg.USE_SECONDARY_LABELS

        # Parse start/end seconds and label vectors upfront
        self._start_secs = [_hms_to_sec(r["start"]) for _, r in df.iterrows()]
        self._end_secs   = [_hms_to_sec(r["end"])   for _, r in df.iterrows()]
        self._labels = np.stack(
            [
                self._parse_window_labels(str(row["primary_label"]))
                for _, row in df.iterrows()
            ],
            axis=0,
        )
        # Per-worker waveform cache (initialised lazily so it works with fork)
        self._cache: Optional[_LRUCache] = None

    def _parse_window_labels(self, raw: str) -> np.ndarray:
        """Parse semicolon-separated species codes from soundscape label."""
        species = [s.strip() for s in raw.split(";") if s.strip()]
        vec = np.zeros(self.n_classes, dtype=np.float32)
        for sp in species:
            if sp in self.label_to_idx:
                vec[self.label_to_idx[sp]] = 1.0
        return vec

    def _get_waveform(self, filepath: str) -> np.ndarray:
        if self._cache is None:
            self._cache = _LRUCache(maxsize=10)
        cached = self._cache.get(filepath)
        if cached is not None:
            return cached
        wav, _ = librosa.load(filepath, sr=self.sr, mono=True)
        self._cache.set(filepath, wav)
        return wav

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = os.path.join(self.soundscapes_dir, str(row["filename"]))
        start_sample = int(self._start_secs[idx] * self.sr)
        end_sample   = int(self._end_secs[idx]   * self.sr)

        try:
            full_wav = self._get_waveform(path)
            window = full_wav[start_sample:end_sample].copy()
            if len(window) == 0:
                raise ValueError("empty window")
        except Exception:
            window = np.zeros(self.n_samples, dtype=np.float32)

        window = self.audio_tf(window)
        img = self.mel_tf(window)
        if self.spec_aug is not None:
            img = self.spec_aug(img)

        return torch.from_numpy(img), torch.from_numpy(self._labels[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Inference dataset: test soundscapes
# ─────────────────────────────────────────────────────────────────────────────

class SoundscapeInferenceDataset(Dataset):
    """
    Enumerates all 5-second windows across every .ogg file in soundscapes_dir.

    Uses soundfile.info for fast duration probing (reads header only).
    Loads audio lazily per window with librosa (offset + duration params).

    Each item: (image_tensor [3, IMG_SIZE, IMG_SIZE], row_id string)
    row_id format: "{stem}_{end_second}"
    """

    def __init__(
        self,
        soundscapes_dir: str,
        audio_transform: AudioTransform,
        mel_transform: MelSpecTransform,
        cfg: CFG,
    ):
        self.audio_tf = audio_transform
        self.mel_tf   = mel_transform
        self.sr       = cfg.SAMPLE_RATE
        self.n_samples = cfg.N_SAMPLES
        self.clip_dur = cfg.CLIP_DURATION

        # Build flat list of (filepath, start_sec, end_sec, row_id)
        self.items: List[Tuple[str, float, float, str]] = []
        ogg_files = sorted(
            f for f in os.listdir(soundscapes_dir)
            if f.lower().endswith(".ogg")
        )
        for fname in ogg_files:
            fpath = os.path.join(soundscapes_dir, fname)
            stem = os.path.splitext(fname)[0]
            try:
                info = sf.info(fpath)
                duration = info.frames / info.samplerate
            except Exception:
                continue

            end = self.clip_dur
            while end <= duration + 1e-3:
                start = end - self.clip_dur
                row_id = f"{stem}_{int(end)}"
                self.items.append((fpath, start, end, row_id))
                end += self.clip_dur

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        fpath, start_sec, end_sec, row_id = self.items[idx]
        try:
            wav, _ = librosa.load(
                fpath,
                sr=self.sr,
                mono=True,
                offset=float(start_sec),
                duration=float(self.clip_dur),
            )
            if len(wav) == 0:
                raise ValueError("empty")
        except Exception:
            wav = np.zeros(self.n_samples, dtype=np.float32)

        wav = self.audio_tf(wav)
        img = self.mel_tf(wav)
        return torch.from_numpy(img), row_id
