"""
BirdCLEF 2026 — Pseudo-label generation script (Stage 2 preparation)

Run after Stage 1 training to generate soft labels for the unlabeled soundscapes:
    python pseudo_label.py

Outputs (in outputs/):
    pseudo_labels_meta.csv   — filename, start_sec, end_sec, max_prob
    pseudo_labels_probs.npy  — float16 array, shape (N_windows, 234)

Only windows where max_prob >= PSEUDO_MIN_CONFIDENCE are saved.
The remaining rows are discarded to keep file sizes manageable.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG, get_device, resolve_paths
from dataset import SoundscapeInferenceDataset, build_species_info
from model import BirdModel
from transforms import AudioTransform, MelSpecTransform


def load_model(ckpt_path: str, cfg: CFG, device: torch.device) -> BirdModel:
    cfg.PRETRAINED = False
    model = BirdModel(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    snd = ckpt.get("snd_cmap", ckpt.get("val_cmap", "?"))
    print(f"  {os.path.basename(ckpt_path)}  snd_cmAP={snd}")
    return model


def get_unlabeled_files(soundscapes_dir: str, labels_csv_path: str) -> list[str]:
    """Return .ogg files in soundscapes_dir that are NOT in the labeled CSV."""
    labeled = set(pd.read_csv(labels_csv_path)["filename"].unique())
    all_files = sorted(f for f in os.listdir(soundscapes_dir) if f.lower().endswith(".ogg"))
    unlabeled = [f for f in all_files if f not in labeled]
    print(f"  Total soundscapes: {len(all_files)}")
    print(f"  Already labeled:   {len(labeled)}")
    print(f"  Unlabeled (will pseudo-label): {len(unlabeled)}")
    return unlabeled


def main() -> None:
    cfg = CFG()
    resolve_paths(cfg)
    device = get_device(cfg)
    print(f"Device: {device}")

    # ── Load Stage 1 ensemble ─────────────────────────────────────────────────
    import glob
    ckpt_paths = sorted(glob.glob(os.path.join(cfg.OUTPUT_DIR, "fold*_best.pth")))
    if not ckpt_paths:
        raise FileNotFoundError(
            "No fold*_best.pth found in outputs/. "
            "Run Stage 1 training first: python train.py --all-folds"
        )
    print(f"\nLoading {len(ckpt_paths)} Stage 1 checkpoint(s):")
    models = [load_model(p, cfg, device) for p in ckpt_paths]

    # ── Identify unlabeled soundscapes ────────────────────────────────────────
    print("\nScanning soundscapes...")
    unlabeled_files = get_unlabeled_files(
        cfg.TRAIN_SOUNDSCAPES_DIR,
        cfg.TRAIN_SOUNDSCAPES_LABELS_CSV,
    )
    if not unlabeled_files:
        print("No unlabeled soundscapes found. Nothing to do.")
        return

    # ── Build inference dataset over unlabeled files only ────────────────────
    # Temporarily point the inference dataset at a filtered view by creating
    # a small wrapper directory listing; simpler: pass a custom file list.
    audio_tf = AudioTransform(cfg, is_train=False)
    mel_tf   = MelSpecTransform(cfg)

    # Re-use SoundscapeInferenceDataset but restrict to unlabeled files.
    # We do this by monkey-patching the items list after construction.
    full_dataset = SoundscapeInferenceDataset(cfg.TRAIN_SOUNDSCAPES_DIR, audio_tf, mel_tf, cfg)

    unlabeled_set = set(unlabeled_files)
    full_dataset.items = [
        item for item in full_dataset.items
        if os.path.basename(item[0]) in unlabeled_set
    ]
    print(f"\nTotal 5-second windows to pseudo-label: {len(full_dataset):,}")

    loader = DataLoader(
        full_dataset,
        batch_size=cfg.INFER_BATCH_SIZE,
        shuffle=False,
        num_workers=0,   # file cache lives in main process
        pin_memory=cfg.PIN_MEMORY and device.type == "cuda",
    )

    # ── Run ensemble inference ────────────────────────────────────────────────
    all_row_ids: list[str] = []
    all_probs_list: list[np.ndarray] = []

    with torch.no_grad():
        for images, row_ids in tqdm(loader, desc="Pseudo-labeling"):
            images = images.to(device)
            batch_sum: np.ndarray | None = None
            for m in models:
                p = torch.sigmoid(m(images)).cpu().float().numpy()
                batch_sum = p if batch_sum is None else batch_sum + p
            avg = batch_sum / len(models)
            all_row_ids.extend(row_ids)
            all_probs_list.append(avg)

    if not all_probs_list:
        print("No windows processed.")
        return

    all_probs = np.concatenate(all_probs_list, axis=0)  # (N, 234)
    max_probs  = all_probs.max(axis=1)                   # (N,)

    # ── Parse row_ids into metadata columns ───────────────────────────────────
    # row_id format: "{stem}_{end_second}"  e.g. "BC2026_Train_0001_S08_..._60"
    filenames, end_secs = [], []
    for row_id in all_row_ids:
        parts     = row_id.rsplit("_", 1)
        end_sec   = int(parts[1])
        fname     = parts[0] + ".ogg"
        filenames.append(fname)
        end_secs.append(end_sec)

    start_secs = [e - int(cfg.CLIP_DURATION) for e in end_secs]

    meta = pd.DataFrame({
        "filename":  filenames,
        "start_sec": start_secs,
        "end_sec":   end_secs,
        "max_prob":  max_probs,
    })

    # ── Filter by confidence threshold ────────────────────────────────────────
    keep_mask = max_probs >= cfg.PSEUDO_MIN_CONFIDENCE
    meta_filtered  = meta[keep_mask].reset_index(drop=True)
    probs_filtered = all_probs[keep_mask]

    n_total  = len(meta)
    n_kept   = len(meta_filtered)
    print(f"\nWindows total:  {n_total:,}")
    print(f"Windows kept (max_prob >= {cfg.PSEUDO_MIN_CONFIDENCE}): {n_kept:,} ({n_kept/n_total*100:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    meta_path  = cfg.PSEUDO_LABELS_META
    probs_path = cfg.PSEUDO_LABELS_PROBS

    meta_filtered.to_csv(meta_path, index=False)
    np.save(probs_path, probs_filtered.astype(np.float16))  # float16 halves file size

    print(f"\nSaved:")
    print(f"  {meta_path}   ({os.path.getsize(meta_path)/1e6:.1f} MB)")
    print(f"  {probs_path}  ({os.path.getsize(probs_path)/1e6:.1f} MB)")
    print("\nNext: python train.py --stage2")


if __name__ == "__main__":
    main()
