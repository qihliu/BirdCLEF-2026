"""
BirdCLEF 2026 — Local Inference Script

Loads all fold checkpoints from outputs/, runs ensemble inference on
test_soundscapes/, and writes submission.csv.

Usage:
    python inference.py
    python inference.py --ckpt outputs/fold0_best.pth  # single checkpoint
"""

import argparse
import glob
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


def load_model(checkpoint_path: str, cfg: CFG, device: torch.device) -> BirdModel:
    model = BirdModel(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    snd_cmap  = ckpt.get("snd_cmap",  ckpt.get("val_cmap", "?"))
    clip_cmap = ckpt.get("clip_cmap", "?")
    print(f"  Loaded {os.path.basename(checkpoint_path)}  "
          f"(snd_cmAP={snd_cmap}, clip_cmAP={clip_cmap})")
    return model


@torch.no_grad()
def run_inference(
    models: list,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    all_row_ids: list = []
    all_probs_sum: np.ndarray | None = None

    for images, row_ids in tqdm(loader, desc="Inference"):
        images = images.to(device, non_blocking=True)
        batch_sum: np.ndarray | None = None
        for m in models:
            logits = m(images)
            probs  = torch.sigmoid(logits).cpu().float().numpy()
            batch_sum = probs if batch_sum is None else batch_sum + probs
        batch_avg = batch_sum / len(models)

        all_row_ids.extend(row_ids)
        if all_probs_sum is None:
            all_probs_sum = batch_avg
        else:
            all_probs_sum = np.concatenate([all_probs_sum, batch_avg], axis=0)

    return all_row_ids, all_probs_sum


def build_submission(
    row_ids: list,
    probs: np.ndarray,
    species_list: list,
    sample_submission_path: str,
) -> pd.DataFrame:
    sub = pd.DataFrame(probs, columns=species_list)
    sub.insert(0, "row_id", row_ids)

    # Reorder columns to exactly match the sample submission
    sample = pd.read_csv(sample_submission_path)
    sub = sub.reindex(columns=sample.columns, fill_value=0.0)
    return sub


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt", nargs="+", default=None,
        help="Explicit checkpoint path(s). Default: all fold*_best.pth in outputs/",
    )
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = CFG()
    resolve_paths(cfg)
    if args.batch_size is not None:
        cfg.INFER_BATCH_SIZE = args.batch_size

    device = get_device(cfg)
    print(f"Device: {device}")

    # Find checkpoints
    if args.ckpt:
        ckpt_paths = args.ckpt
    else:
        ckpt_paths = sorted(glob.glob(os.path.join(cfg.OUTPUT_DIR, "fold*_best.pth")))

    if not ckpt_paths:
        print("No checkpoints found. Train first with: python train.py")
        return

    print(f"\nLoading {len(ckpt_paths)} checkpoint(s):")
    models = [load_model(p, cfg, device) for p in ckpt_paths]

    species_info = build_species_info(cfg.TAXONOMY_CSV)
    species_list = species_info["species_list"]

    # Check if test soundscapes exist (they are hidden on Kaggle)
    test_dir = cfg.TEST_SOUNDSCAPES_DIR
    ogg_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".ogg")]
    if not ogg_files:
        print(
            "\nNo .ogg files found in test_soundscapes/.\n"
            "Locally the test set is hidden; on Kaggle it is populated at submission time.\n"
            "Copying sample_submission.csv as output for format verification."
        )
        sample = pd.read_csv(cfg.SAMPLE_SUBMISSION_CSV)
        out_path = os.path.join(cfg.OUTPUT_DIR, "submission.csv")
        sample.to_csv(out_path, index=False)
        print(f"Written: {out_path}")
        return

    print(f"\nFound {len(ogg_files)} test soundscape(s).")

    audio_tf = AudioTransform(cfg, is_train=False)
    mel_tf   = MelSpecTransform(cfg)
    dataset  = SoundscapeInferenceDataset(test_dir, audio_tf, mel_tf, cfg)
    loader   = DataLoader(
        dataset,
        batch_size=cfg.INFER_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY and device.type == "cuda",
    )
    print(f"Total windows to predict: {len(dataset)}\n")

    row_ids, probs = run_inference(models, loader, device)

    sub = build_submission(row_ids, probs, species_list, cfg.SAMPLE_SUBMISSION_CSV)
    out_path = os.path.join(cfg.OUTPUT_DIR, "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nsubmission.csv written → {out_path}")
    print(f"Shape: {sub.shape}")
    print(sub.head(3).to_string())


if __name__ == "__main__":
    main()
