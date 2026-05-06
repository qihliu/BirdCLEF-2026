"""
BirdCLEF 2026 — Training Script

Usage:
    python train.py                     # train fold 0 only (quick start)
    python train.py --folds 0 1 2 3 4  # train all folds
    python train.py --min-rating 3.0   # use only high-quality clips
"""

import argparse
import math
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import CFG, get_device, resolve_paths
from dataset import (BirdTrainDataset, PseudoLabeledDataset,
                     SoundscapeTrainDataset, build_species_info,
                     compute_pos_weights, compute_sample_weights,
                     split_soundscape_labels)
from model import BirdModel
from transforms import (AudioTransform, MelSpecTransform, SpecAugment,
                         mixup_criterion, mixup_data)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Folds
# ─────────────────────────────────────────────────────────────────────────────

def create_folds(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    """Add a 'fold' column via StratifiedKFold on primary_label."""
    df = df.copy()
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df["primary_label"].astype(str))):
        df.loc[val_idx, "fold"] = fold
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer & scheduler
# ─────────────────────────────────────────────────────────────────────────────

def get_optimizer(model: nn.Module, cfg: CFG) -> torch.optim.Optimizer:
    # No weight decay on bias and LayerNorm/BN parameters
    no_decay = {"bias", "bn", "norm"}
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if any(nd in name.lower() for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": cfg.WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.LR,
    )


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: CFG,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps  = cfg.EPOCHS * steps_per_epoch
    warmup_steps = cfg.WARMUP_EPOCHS * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(cfg.MIN_LR / cfg.LR, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-sample weighted BCE loss supporting soft labels.

    labels may contain values in (0, 1) — soft labels from pseudo-labeling.
    weights are per-sample scalars:
      - 1.0  for hard-labeled clips / soundscape windows
      - confidence * PSEUDO_LOSS_WEIGHT  for pseudo-labeled windows

    Shape: logits / labels (B, C), weights (B,) → scalar loss.
    """
    per_elem = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight, reduction="none"
    )                                           # (B, C)
    per_sample = per_elem.mean(dim=1)           # (B,)
    return (per_sample * weights).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Metric: approximate padded cmAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_padded_cmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    padding: int = 5,
) -> float:
    """
    Compute padded class-mean Average Precision.

    For each class, prepend `padding` true-positive rows (score=1.0) before
    computing AP. This mimics the official competition metric.
    """
    n_classes = y_true.shape[1]
    aps = []
    pad_true = np.ones(padding, dtype=np.float32)
    pad_pred = np.ones(padding, dtype=np.float32)
    for c in range(n_classes):
        t = np.concatenate([y_true[:, c], pad_true])
        p = np.concatenate([y_pred[:, c], pad_pred])
        if t.sum() == padding:
            # No real positives in the validation set for this class
            ap = 0.0
        else:
            ap = average_precision_score(t, p)
        aps.append(ap)
    return float(np.mean(aps))


# ─────────────────────────────────────────────────────────────────────────────
# Train / validate epochs
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: Optional[torch.Tensor],   # per-class pos_weight tensor or None
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    cfg: CFG,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
) -> float:
    """
    Each batch is a 3-tuple (images, labels, weights).
    labels may be soft (float in [0,1]) for pseudo-labeled samples.
    weights are per-sample: 1.0 for labeled, confidence*PSEUDO_LOSS_WEIGHT for pseudo.
    """
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for step, (images, labels, weights) in enumerate(pbar):
        images  = images.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        # Mixup (applied to both images and labels; works for soft labels too)
        do_mixup = cfg.MIXUP_PROB > 0 and random.random() < cfg.MIXUP_PROB
        if do_mixup:
            images, y_a, y_b, lam = mixup_data(images, labels, cfg.MIXUP_ALPHA)

        def compute_loss(logits):
            if do_mixup:
                # Mixup: blend two weighted losses
                la = weighted_bce_loss(logits, y_a, weights, pos_weight)
                lb = weighted_bce_loss(logits, y_b, weights, pos_weight)
                return lam * la + (1 - lam) * lb
            return weighted_bce_loss(logits, labels, weights, pos_weight)

        if use_amp:
            with torch.cuda.amp.autocast():
                loss = compute_loss(model(images))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = compute_loss(model(images))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        optimizer.zero_grad()
        scheduler.step()

        total_loss += loss.item()
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss  = 0.0
    all_logits  = []
    all_labels  = []

    pbar = tqdm(loader, desc="             [val]  ", leave=False)
    for images, labels, _ in pbar:   # weight ignored during validation
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.cpu().float())
        all_labels.append(labels.cpu().float())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    y_pred = 1.0 / (1.0 + np.exp(-all_logits))   # sigmoid

    val_loss = total_loss / max(1, len(loader))
    val_cmap = compute_padded_cmap(all_labels, y_pred)
    return val_loss, val_cmap


# ─────────────────────────────────────────────────────────────────────────────
# Fold runner
# ─────────────────────────────────────────────────────────────────────────────

def run_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    soundscape_df_path: str,
    cfg: CFG,
    species_info: dict,
    device: torch.device,
    resume_ckpt: Optional[str] = None,   # path to Stage 1 checkpoint for warm start
    pseudo_meta: Optional[str] = None,   # path to pseudo_labels_meta.csv
    pseudo_probs: Optional[str] = None,  # path to pseudo_labels_probs.npy
) -> str:
    print(f"\n{'='*60}")
    print(f"  FOLD {fold}  |  train={len(train_df)}  val={len(val_df)}")
    print(f"{'='*60}")

    train_audio_tf = AudioTransform(cfg, is_train=True)
    val_audio_tf   = AudioTransform(cfg, is_train=False)
    mel_tf         = MelSpecTransform(cfg)
    spec_aug       = SpecAugment(cfg)

    # ── Soundscape train/val split (file-level, fixed across folds) ──────────
    snd_train_df, snd_val_df = split_soundscape_labels(
        soundscape_df_path,
        val_fraction=cfg.SOUNDSCAPE_VAL_FRACTION,
        seed=cfg.SEED,
    )
    print(f"  Soundscape split: {len(snd_train_df)} train windows "
          f"/ {len(snd_val_df)} val windows "
          f"(from {snd_val_df['filename'].nunique()} held-out files)")

    # ── Datasets ──────────────────────────────────────────────────────────────
    bird_train_ds = BirdTrainDataset(
        train_df, cfg.TRAIN_AUDIO_DIR, species_info,
        train_audio_tf, mel_tf, spec_aug, cfg,
    )
    snd_train_ds = SoundscapeTrainDataset(
        snd_train_df, cfg.TRAIN_SOUNDSCAPES_DIR, species_info,
        train_audio_tf, mel_tf, spec_aug, cfg,
    )
    train_ds = ConcatDataset([bird_train_ds, snd_train_ds])

    # Clip validation: all 234 species, stable signal for learning-curve tracking
    clip_val_ds = BirdTrainDataset(
        val_df, cfg.TRAIN_AUDIO_DIR, species_info,
        val_audio_tf, mel_tf, None, cfg,
    )
    # Soundscape validation: domain-matched to test set, used for checkpoint selection
    snd_val_ds = SoundscapeTrainDataset(
        snd_val_df, cfg.TRAIN_SOUNDSCAPES_DIR, species_info,
        val_audio_tf, mel_tf, None, cfg,
    )

    # ── Initial sampler & loaders (may be rebuilt in Stage 2 block below) ─────
    if cfg.USE_WEIGHTED_SAMPLER:
        _sw = compute_sample_weights(train_df, len(snd_train_ds), power=cfg.SAMPLER_POWER)
        sampler = WeightedRandomSampler(_sw, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY and device.type == "cuda", drop_last=True,
        )
        print(f"  WeightedRandomSampler enabled (power={cfg.SAMPLER_POWER})")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY and device.type == "cuda", drop_last=True,
        )

    clip_val_loader = DataLoader(
        clip_val_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY and device.type == "cuda",
    )
    snd_val_loader = DataLoader(
        snd_val_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY and device.type == "cuda",
    )

    # ── Stage 2: add pseudo-labeled soundscape windows to training ────────────
    stage2 = pseudo_meta is not None and pseudo_probs is not None
    if stage2:
        pseudo_ds = PseudoLabeledDataset(
            pseudo_meta, pseudo_probs,
            cfg.TRAIN_SOUNDSCAPES_DIR,
            train_audio_tf, mel_tf, cfg,
        )
        train_ds = ConcatDataset([train_ds, pseudo_ds])
        print(f"  Stage 2: added {len(pseudo_ds):,} pseudo-labeled windows "
              f"(min_confidence={cfg.PSEUDO_MIN_CONFIDENCE}, "
              f"loss_weight={cfg.PSEUDO_LOSS_WEIGHT})")

    # Rebuild sampler with updated dataset size
    if cfg.USE_WEIGHTED_SAMPLER:
        n_pseudo = len(pseudo_ds) if stage2 else 0
        sample_weights = compute_sample_weights(
            train_df, len(snd_train_ds) + n_pseudo, power=cfg.SAMPLER_POWER
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.BATCH_SIZE,
            sampler=sampler,
            num_workers=cfg.NUM_WORKERS if not stage2 else 0,
            pin_memory=cfg.PIN_MEMORY and device.type == "cuda",
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS if not stage2 else 0,
            pin_memory=cfg.PIN_MEMORY and device.type == "cuda",
            drop_last=True,
        )

    # ── Loss (pos_weight for class imbalance, passed to weighted_bce_loss) ───
    if cfg.USE_POS_WEIGHT:
        pos_weight_tensor = compute_pos_weights(
            train_df, snd_train_df, species_info, max_weight=cfg.POS_WEIGHT_MAX
        ).to(device)
        print(f"  pos_weight enabled (max={cfg.POS_WEIGHT_MAX}, "
              f"mean={pos_weight_tensor.mean().item():.2f})")
    else:
        pos_weight_tensor = None

    # Unweighted criterion used only for validation (fair comparison)
    val_criterion = nn.BCEWithLogitsLoss()

    model = BirdModel(cfg).to(device)

    # ── Warm-start from Stage 1 checkpoint ───────────────────────────────────
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Resumed from {os.path.basename(resume_ckpt)}")

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    suffix = "_stage2" if stage2 else ""
    best_ckpt_metric = -1.0
    best_ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"fold{fold}_best{suffix}.pth")

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, pos_weight_tensor, scheduler,
            device, cfg, scaler, epoch,
        )
        clip_val_loss, clip_cmap = validate_one_epoch(model, clip_val_loader, val_criterion, device)
        _,             snd_cmap  = validate_one_epoch(model, snd_val_loader,  val_criterion, device)
        elapsed = time.time() - t0

        # Primary checkpoint metric: soundscape_cmAP (domain-matched to test)
        # Secondary: clip_cmAP (stable learning-curve signal over all 234 species)
        ckpt_metric = snd_cmap if cfg.CKPT_METRIC == "soundscape_cmap" else clip_cmap

        print(
            f"Epoch {epoch:02d}/{cfg.EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"clip_val_loss={clip_val_loss:.4f} | "
            f"clip_cmAP={clip_cmap:.4f} | "
            f"snd_cmAP={snd_cmap:.4f}* | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"time={elapsed:.0f}s"
        )

        if ckpt_metric > best_ckpt_metric:
            best_ckpt_metric = ckpt_metric
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "clip_val_loss": clip_val_loss,
                    "clip_cmap": clip_cmap,
                    "snd_cmap": snd_cmap,
                    "ckpt_metric": cfg.CKPT_METRIC,
                    "stage2": stage2,
                    "cfg": cfg.__dict__.copy(),
                },
                best_ckpt_path,
            )
            print(f"  ✓ saved best checkpoint "
                  f"(snd_cmAP={snd_cmap:.4f}, clip_cmAP={clip_cmap:.4f})")

    print(f"\nFold {fold} done. Best {cfg.CKPT_METRIC} = {best_ckpt_metric:.4f}"
          f"  →  {best_ckpt_path}")
    return best_ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BirdCLEF 2026 training. Stage 1: no flags. Stage 2: --stage2."
    )
    p.add_argument("--folds",      nargs="+", type=int, default=None)
    p.add_argument("--all-folds",  action="store_true", help="Train all 5 folds")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--min-rating", type=float, default=None)

    # Stage 2 pseudo-label options
    p.add_argument("--stage2",     action="store_true",
                   help="Include pseudo-labeled soundscapes in training")
    p.add_argument("--resume",     type=str, default=None,
                   help="Checkpoint to warm-start from (e.g. outputs/fold0_best.pth). "
                        "Defaults to the matching fold's Stage 1 checkpoint when --stage2.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = CFG()
    resolve_paths(cfg)

    if args.epochs     is not None: cfg.EPOCHS    = args.epochs
    if args.batch_size is not None: cfg.BATCH_SIZE = args.batch_size
    if args.min_rating is not None: cfg.MIN_RATING = args.min_rating

    if args.all_folds:
        folds_to_train = list(range(cfg.N_FOLDS))
    elif args.folds is not None:
        folds_to_train = args.folds
    else:
        folds_to_train = [0]

    seed_everything(cfg.SEED)
    device = get_device(cfg)

    stage_label = "Stage 2 (pseudo-label)" if args.stage2 else "Stage 1 (labeled only)"
    print(f"Device: {device}  |  {stage_label}")
    print(f"Training folds: {folds_to_train}  |  Epochs: {cfg.EPOCHS}  |  Batch: {cfg.BATCH_SIZE}")

    # Validate pseudo-label files exist before starting
    if args.stage2:
        for path, name in [(cfg.PSEUDO_LABELS_META, "meta CSV"),
                           (cfg.PSEUDO_LABELS_PROBS, "probs npy")]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Pseudo-label {name} not found at {path}. "
                    "Run: python pseudo_label.py"
                )
        print(f"  Pseudo-label files: {cfg.PSEUDO_LABELS_META}")

    species_info = build_species_info(cfg.TAXONOMY_CSV)
    print(f"Species: {len(species_info['species_list'])}")

    df = pd.read_csv(cfg.TRAIN_CSV)
    if cfg.MIN_RATING > 0:
        df = df[df["rating"] >= cfg.MIN_RATING]
    df = create_folds(df, cfg)
    print(f"Labeled clips (rating >= {cfg.MIN_RATING}): {len(df)}")

    for fold in folds_to_train:
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df   = df[df["fold"] == fold].reset_index(drop=True)

        # For Stage 2, default to the matching Stage 1 checkpoint as warm start
        resume = args.resume
        if args.stage2 and resume is None:
            auto = os.path.join(cfg.OUTPUT_DIR, f"fold{fold}_best.pth")
            if os.path.exists(auto):
                resume = auto
                print(f"  Auto-resume: {auto}")
            else:
                print(f"  Warning: no Stage 1 checkpoint found for fold {fold} — cold start")

        run_fold(
            fold, train_df, val_df,
            cfg.TRAIN_SOUNDSCAPES_LABELS_CSV,
            cfg, species_info, device,
            resume_ckpt=resume,
            pseudo_meta=cfg.PSEUDO_LABELS_META  if args.stage2 else None,
            pseudo_probs=cfg.PSEUDO_LABELS_PROBS if args.stage2 else None,
        )

    print("\nAll requested folds complete.")


if __name__ == "__main__":
    main()
