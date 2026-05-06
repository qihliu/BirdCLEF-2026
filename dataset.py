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


# ─────────────────────────────────────────────────────────────────────────────
# Class-imbalance utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_sample_weights(
    bird_df: pd.DataFrame,
    n_soundscape_windows: int,
    power: float = 0.5,
) -> torch.Tensor:
    """
    Compute per-sample weights for WeightedRandomSampler.

    Bird clips: weight = 1 / count(primary_label) ^ power
      power=0.5 (default) is the square-root schedule — a good middle ground
      between uniform sampling (power=0) and full inverse-frequency (power=1).
      Full inverse-frequency oversamples 1-clip species 499x, risking severe
      overfitting; square-root dampens this to ~22x.

    Soundscape windows: assigned the mean weight of bird clips so they
      contribute neutrally (domain-adaptation signal, not imbalance correction).
    """
    counts = bird_df["primary_label"].astype(str).value_counts()
    labels = bird_df["primary_label"].astype(str).values
    raw_w  = np.array(
        [1.0 / (float(counts.get(lbl, 1)) ** power) for lbl in labels],
        dtype=np.float32,
    )
    bird_weights = torch.from_numpy(raw_w)
    mean_w = bird_weights.mean().item()
    snd_weights = torch.full((n_soundscape_windows,), mean_w, dtype=torch.float32)
    return torch.cat([bird_weights, snd_weights])


def compute_pos_weights(
    bird_df: pd.DataFrame,
    ssl_df: pd.DataFrame,
    species_info: Dict,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    Compute positive-class weights for BCEWithLogitsLoss.

    For each class c:
        pos_weight[c] = n_negative[c] / n_positive[c]

    This penalises false negatives for rare classes proportionally.
    Capped at max_weight (default 10) to prevent a single rare-species
    false-negative from dominating the entire loss batch.

    Without the cap, a species with 1 clip would get pos_weight ≈ 35,548,
    making training numerically unstable.
    """
    label_to_idx = species_info["label_to_idx"]
    n_classes = len(label_to_idx)
    pos_counts = np.zeros(n_classes, dtype=np.float64)

    # Positives from train_audio clips — primary labels (vectorised)
    for lbl, cnt in bird_df["primary_label"].astype(str).value_counts().items():
        idx = label_to_idx.get(lbl)
        if idx is not None:
            pos_counts[idx] += float(cnt)
    # Secondary labels (need per-row parsing, but iterate the raw column only)
    for raw in bird_df["secondary_labels"]:
        for sec in _parse_secondary(str(raw)):
            sidx = label_to_idx.get(sec)
            if sidx is not None:
                pos_counts[sidx] += 1.0

    # Positives from labeled soundscape windows
    for _, row in ssl_df.iterrows():
        for sp in str(row["primary_label"]).split(";"):
            idx = label_to_idx.get(sp.strip())
            if idx is not None:
                pos_counts[idx] += 1.0

    n_total = len(bird_df) + len(ssl_df)
    neg_counts = n_total - pos_counts
    # Avoid div-by-zero for classes with no positives → assign max weight
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(pos_counts > 0, neg_counts / pos_counts, max_weight)
    weights = np.clip(weights, 1.0, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


def _hms_to_sec(hms: str) -> int:
    """Convert 'HH:MM:SS' to integer seconds."""
    h, m, s = hms.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def split_soundscape_labels(
    labels_csv_path: str,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    File-level train/val split of labeled soundscape windows.

    Split is on unique *filenames*, not individual windows, to prevent
    data leakage: consecutive 5-second windows from the same recording
    share the same acoustic environment and are highly correlated.

    Returns:
        (train_df, val_df) — both deduplicated on (filename, start, end).
    """
    df = pd.read_csv(labels_csv_path).drop_duplicates(
        subset=["filename", "start", "end"]
    ).reset_index(drop=True)

    files = df["filename"].unique()
    rng = np.random.RandomState(seed)
    n_val = max(1, int(len(files) * val_fraction))
    val_files = set(rng.choice(files, size=n_val, replace=False))

    train_df = df[~df["filename"].isin(val_files)].reset_index(drop=True)
    val_df   = df[ df["filename"].isin(val_files)].reset_index(drop=True)
    return train_df, val_df


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
        self.sr = cfg.SAMPLE_RATE

        # Precompute secondary label lists and label vectors.
        # itertuples() is ~10x faster than iterrows() for large DataFrames.
        self._secondary = [
            _parse_secondary(str(row.secondary_labels))
            for row in self.df.itertuples(index=False)
        ]
        self._labels = np.stack(
            [
                build_multilabel_vector(
                    str(row.primary_label),
                    self._secondary[i],
                    self.label_to_idx,
                    self.n_classes,
                    self.use_secondary,
                )
                for i, row in enumerate(self.df.itertuples(index=False))
            ],
            axis=0,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = os.path.join(self.audio_base_dir, str(row["filename"]))

        try:
            waveform, _ = librosa.load(path, sr=self.sr, mono=True)
            if len(waveform) == 0:
                raise ValueError("empty audio")
        except Exception:
            waveform = np.zeros(self.audio_tf.n_samples, dtype=np.float32)

        waveform = self.audio_tf(waveform)
        img = self.mel_tf(waveform)
        if self.spec_aug is not None:
            img = self.spec_aug(img)

        label = self._labels[idx]
        return torch.from_numpy(img), torch.from_numpy(label), torch.tensor(1.0)


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

    Accepts either a CSV path (reads + deduplicates) or a pre-filtered
    DataFrame (e.g. from split_soundscape_labels) so it can be reused
    for both training and soundscape validation without duplication.

    Uses a per-worker LRU cache to avoid reloading the same soundscape file
    multiple times within one epoch.
    """

    def __init__(
        self,
        labels_df_or_path,            # pd.DataFrame or str path to CSV
        soundscapes_dir: str,
        species_info: Dict,
        audio_transform: AudioTransform,
        mel_transform: MelSpecTransform,
        spec_augment: Optional[SpecAugment],
        cfg: CFG,
    ):
        if isinstance(labels_df_or_path, pd.DataFrame):
            df = labels_df_or_path.reset_index(drop=True)
        else:
            df = pd.read_csv(labels_df_or_path)
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

        return torch.from_numpy(img), torch.from_numpy(self._labels[idx]), torch.tensor(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Inference dataset: test soundscapes
# ─────────────────────────────────────────────────────────────────────────────

class SoundscapeInferenceDataset(Dataset):
    """
    Enumerates all 5-second windows across every .ogg file in soundscapes_dir.

    Each soundscape is loaded ONCE (on first access) and cached for the
    lifetime of the dataset. Windows are sliced from the in-memory waveform,
    so each file is read from disk exactly once regardless of how many windows
    it contains.

    IMPORTANT: use num_workers=0 in the DataLoader. With workers > 0 each
    worker gets its own copy of the cache, breaking the one-load-per-file
    guarantee and causing redundant I/O.

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
        self.audio_tf  = audio_transform
        self.mel_tf    = mel_transform
        self.sr        = cfg.SAMPLE_RATE
        self.n_samples = cfg.N_SAMPLES
        self.clip_dur  = cfg.CLIP_DURATION

        # One-entry file cache: stores the last-loaded (fpath, waveform) pair.
        # With shuffle=False and items ordered by file this gives a 100% hit
        # rate — each file is loaded exactly once.
        self._cached_fpath: Optional[str] = None
        self._cached_wav:   Optional[np.ndarray] = None

        # Build flat list of (filepath, start_sample, end_sample, row_id).
        # Sample offsets are pre-computed so __getitem__ only slices memory.
        self.items: List[Tuple[str, int, int, str]] = []
        ogg_files = sorted(
            f for f in os.listdir(soundscapes_dir)
            if f.lower().endswith(".ogg")
        )
        for fname in ogg_files:
            fpath = os.path.join(soundscapes_dir, fname)
            stem  = os.path.splitext(fname)[0]
            try:
                info     = sf.info(fpath)
                duration = info.frames / info.samplerate
            except Exception:
                continue

            end = self.clip_dur
            while end <= duration + 1e-3:
                start_sample = int((end - self.clip_dur) * self.sr)
                end_sample   = int(end * self.sr)
                row_id       = f"{stem}_{int(end)}"
                self.items.append((fpath, start_sample, end_sample, row_id))
                end += self.clip_dur

    def _load_file(self, fpath: str) -> np.ndarray:
        if fpath != self._cached_fpath:
            wav, _ = librosa.load(fpath, sr=self.sr, mono=True)
            self._cached_fpath = fpath
            self._cached_wav   = wav
        return self._cached_wav

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        fpath, start_sample, end_sample, row_id = self.items[idx]
        try:
            wav    = self._load_file(fpath)
            window = wav[start_sample:end_sample].copy()
            if len(window) == 0:
                raise ValueError("empty")
        except Exception:
            window = np.zeros(self.n_samples, dtype=np.float32)

        wav = self.audio_tf(window)
        img = self.mel_tf(wav)
        return torch.from_numpy(img), row_id


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-labeled dataset (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

class PseudoLabeledDataset(Dataset):
    """
    Soundscape windows with soft pseudo-labels produced by the Stage 1 model.

    Soft labels are the raw model probabilities (float16 → float32), not 0/1.
    Per-class values below PSEUDO_LABEL_THRESHOLD are zeroed out to reduce
    noise from near-zero predictions.

    Each item returns (image, soft_label, sample_weight) where:
      soft_label     — float32 vector in [0, 1], shape (NUM_CLASSES,)
      sample_weight  — max_prob of the window × PSEUDO_LOSS_WEIGHT
                       (higher confidence → larger gradient contribution)

    Use num_workers=0 in the DataLoader so the file cache works correctly.
    """

    def __init__(
        self,
        meta_csv_path: str,
        probs_npy_path: str,
        soundscapes_dir: str,
        audio_transform: AudioTransform,
        mel_transform: MelSpecTransform,
        cfg: CFG,
    ):
        meta = pd.read_csv(meta_csv_path)
        probs = np.load(probs_npy_path).astype(np.float32)  # (N, NUM_CLASSES)

        assert len(meta) == len(probs), "meta and probs row counts must match"

        # Filter by confidence
        keep = meta["max_prob"].values >= cfg.PSEUDO_MIN_CONFIDENCE
        meta  = meta[keep].reset_index(drop=True)
        probs = probs[keep]

        # Zero out per-class probabilities below label threshold (noise reduction)
        probs[probs < cfg.PSEUDO_LABEL_THRESHOLD] = 0.0

        self.soundscapes_dir = soundscapes_dir
        self.audio_tf  = audio_transform
        self.mel_tf    = mel_transform
        self.sr        = cfg.SAMPLE_RATE
        self.n_samples = cfg.N_SAMPLES
        self.pseudo_loss_weight = cfg.PSEUDO_LOSS_WEIGHT

        # Pre-compute sample offsets and weights
        self._fpaths       = [
            os.path.join(soundscapes_dir, fn)
            for fn in meta["filename"].tolist()
        ]
        self._start_samples = (meta["start_sec"].values * self.sr).astype(int)
        self._end_samples   = (meta["end_sec"].values   * self.sr).astype(int)
        self._soft_labels   = probs                              # (N, NUM_CLASSES)
        self._weights       = (meta["max_prob"].values * self.pseudo_loss_weight).astype(np.float32)

        # One-entry file cache (same pattern as SoundscapeInferenceDataset)
        self._cached_fpath: Optional[str] = None
        self._cached_wav:   Optional[np.ndarray] = None

    def _load_file(self, fpath: str) -> np.ndarray:
        if fpath != self._cached_fpath:
            wav, _ = librosa.load(fpath, sr=self.sr, mono=True)
            self._cached_fpath = fpath
            self._cached_wav   = wav
        return self._cached_wav

    def __len__(self) -> int:
        return len(self._fpaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fpath = self._fpaths[idx]
        try:
            wav    = self._load_file(fpath)
            window = wav[self._start_samples[idx]:self._end_samples[idx]].copy()
            if len(window) == 0:
                raise ValueError("empty")
        except Exception:
            window = np.zeros(self.n_samples, dtype=np.float32)

        window = self.audio_tf(window)
        img    = self.mel_tf(window)

        soft_label = self._soft_labels[idx]          # float32, (NUM_CLASSES,)
        weight     = float(self._weights[idx])

        return (
            torch.from_numpy(img),
            torch.from_numpy(soft_label),
            torch.tensor(weight),
        )
