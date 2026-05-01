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
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import CFG, get_device, resolve_paths
from dataset import (BirdTrainDataset, SoundscapeTrainDataset,
                     build_species_info, compute_pos_weights,
                     compute_sample_weights)
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
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    cfg: CFG,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup
        do_mixup = cfg.MIXUP_PROB > 0 and random.random() < cfg.MIXUP_PROB
        if do_mixup:
            images, y_a, y_b, lam = mixup_data(images, labels, cfg.MIXUP_ALPHA)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = (
                    mixup_criterion(criterion, logits, y_a, y_b, lam)
                    if do_mixup
                    else criterion(logits, labels)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = (
                mixup_criterion(criterion, logits, y_a, y_b, lam)
                if do_mixup
                else criterion(logits, labels)
            )
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
    for images, labels in pbar:
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
) -> str:
    print(f"\n{'='*60}")
    print(f"  FOLD {fold}  |  train={len(train_df)}  val={len(val_df)}")
    print(f"{'='*60}")

    train_audio_tf = AudioTransform(cfg, is_train=True)
    val_audio_tf   = AudioTransform(cfg, is_train=False)
    mel_tf         = MelSpecTransform(cfg)
    spec_aug       = SpecAugment(cfg)

    # Training data: clip dataset + soundscape windows (always in train)
    bird_train_ds = BirdTrainDataset(
        train_df, cfg.TRAIN_AUDIO_DIR, species_info,
        train_audio_tf, mel_tf, spec_aug, cfg,
    )
    snd_train_ds = SoundscapeTrainDataset(
        soundscape_df_path, cfg.TRAIN_SOUNDSCAPES_DIR, species_info,
        train_audio_tf, mel_tf, spec_aug, cfg,
    )
    train_ds = ConcatDataset([bird_train_ds, snd_train_ds])

    val_ds = BirdTrainDataset(
        val_df, cfg.TRAIN_AUDIO_DIR, species_info,
        val_audio_tf, mel_tf, None, cfg,
    )

    # ── Weighted sampler (class-imbalance) ────────────────────────────────────
    if cfg.USE_WEIGHTED_SAMPLER:
        ssl_df = pd.read_csv(soundscape_df_path).drop_duplicates(
            subset=["filename", "start", "end"]
        )
        sample_weights = compute_sample_weights(
            train_df, len(snd_train_ds), power=cfg.SAMPLER_POWER
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.BATCH_SIZE,
            sampler=sampler,          # mutually exclusive with shuffle=True
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY and device.type == "cuda",
            drop_last=True,
        )
        print(f"  WeightedRandomSampler enabled (power={cfg.SAMPLER_POWER})")
    else:
        ssl_df = None
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY and device.type == "cuda",
            drop_last=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY and device.type == "cuda",
    )

    # ── Loss with optional pos_weight (class-imbalance) ───────────────────────
    if cfg.USE_POS_WEIGHT:
        if ssl_df is None:
            ssl_df = pd.read_csv(soundscape_df_path).drop_duplicates(
                subset=["filename", "start", "end"]
            )
        pos_weight = compute_pos_weights(
            train_df, ssl_df, species_info, max_weight=cfg.POS_WEIGHT_MAX
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  pos_weight enabled (max={cfg.POS_WEIGHT_MAX}, "
              f"mean={pos_weight.mean().item():.2f})")
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_cmap = -1.0
    best_ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"fold{fold}_best.pth")

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler,
            device, cfg, scaler, epoch,
        )
        val_loss, val_cmap = validate_one_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{cfg.EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_cmAP={val_cmap:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"time={elapsed:.0f}s"
        )

        if val_cmap > best_cmap:
            best_cmap = val_cmap
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_cmap": val_cmap,
                    "cfg": cfg.__dict__.copy(),
                },
                best_ckpt_path,
            )
            print(f"  ✓ saved best checkpoint (cmAP={val_cmap:.4f})")

    print(f"\nFold {fold} done. Best cmAP = {best_cmap:.4f}  →  {best_ckpt_path}")
    return best_ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--folds",      nargs="+", type=int, default=None,
                   help="Which folds to train (default: fold 0 only)")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--min-rating", type=float, default=None)
    p.add_argument("--all-folds",  action="store_true",
                   help="Train all 5 folds (overrides --folds)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = CFG()
    resolve_paths(cfg)

    # CLI overrides
    if args.epochs     is not None: cfg.EPOCHS      = args.epochs
    if args.batch_size is not None: cfg.BATCH_SIZE   = args.batch_size
    if args.min_rating is not None: cfg.MIN_RATING   = args.min_rating
    if args.all_folds:
        folds_to_train = list(range(cfg.N_FOLDS))
    elif args.folds is not None:
        folds_to_train = args.folds
    else:
        folds_to_train = [0]         # default: fold 0 only

    seed_everything(cfg.SEED)
    device = get_device(cfg)
    print(f"Device: {device}")
    print(f"Training folds: {folds_to_train}")
    print(f"Epochs: {cfg.EPOCHS}  |  Batch size: {cfg.BATCH_SIZE}")

    species_info = build_species_info(cfg.TAXONOMY_CSV)
    print(f"Species count: {len(species_info['species_list'])}")

    df = pd.read_csv(cfg.TRAIN_CSV)
    if cfg.MIN_RATING > 0:
        df = df[df["rating"] >= cfg.MIN_RATING]
    df = create_folds(df, cfg)
    print(f"Training samples (rating >= {cfg.MIN_RATING}): {len(df)}")

    for fold in folds_to_train:
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df   = df[df["fold"] == fold].reset_index(drop=True)
        run_fold(
            fold, train_df, val_df,
            cfg.TRAIN_SOUNDSCAPES_LABELS_CSV,
            cfg, species_info, device,
        )

    print("\nAll requested folds complete.")


if __name__ == "__main__":
    main()
