"""
train_diffusion.py – Training script for the DDPM teacher model.

Usage
-----
    # Quick test on 2 subjects, 30 epochs
    python train_diffusion.py --subjects 1 2 --epochs 30 --snr 10

    # Full overnight run on all 9 subjects
    python train_diffusion.py --subjects 1 2 3 4 5 6 7 8 9 --epochs 100
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Allow running from any directory
sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import (
    load_bci_competition_data,
    preprocess_pipeline,
    create_train_val_test_split,
    inject_noise,
)
from diffusion import DiffusionConfig, GaussianDiffusion, UNet1D, train_diffusion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('train_diffusion')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train DDPM teacher for EEG denoising')
    p.add_argument('--subjects', type=int, nargs='+', default=[1, 2],
                   help='Subject IDs to include (default: 1 2)')
    p.add_argument('--epochs', type=int, default=30,
                   help='Training epochs (default: 30)')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--snr', type=int, default=10,
                   help='SNR (dB) of injected noise used as model input')
    p.add_argument('--n_steps', type=int, default=1000,
                   help='Diffusion timesteps T (default: 1000)')
    p.add_argument('--schedule', choices=['linear', 'cosine'], default='linear')
    p.add_argument('--checkpoint_dir', default='models/diffusion_teacher')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--early_stop', type=int, default=20,
                   help='Early-stop patience epochs')
    p.add_argument('--data_dir', default='./data')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helper
# ─────────────────────────────────────────────────────────────────────────────

def build_channel_dataset(
    X_clean: np.ndarray,
    X_noisy: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Flatten (trials, channels, 750) → (trials * channels, 1, 750) and
    return a DataLoader of (noisy, clean) pairs.
    """
    n_trials, n_ch, n_samp = X_clean.shape
    # Reshape: each channel window becomes an independent sample
    clean_flat = X_clean.reshape(n_trials * n_ch, 1, n_samp).astype(np.float32)
    noisy_flat = X_noisy.reshape(n_trials * n_ch, 1, n_samp).astype(np.float32)

    ds = TensorDataset(
        torch.from_numpy(noisy_flat),
        torch.from_numpy(clean_flat),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=False, num_workers=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("=== Phase 3: Training Diffusion Teacher ===")
    logger.info("Subjects : %s", args.subjects)
    logger.info("Epochs   : %d", args.epochs)
    logger.info("Noise SNR: %d dB", args.snr)

    # ── 1. Load + preprocess data ────────────────────────────────────────────
    logger.info("Loading BCI dataset …")
    data = load_bci_competition_data(subjects=args.subjects,
                                     cache_dir=args.data_dir)
    X_raw = data['X']        # (total_trials, 22, 750)
    y = data['y']
    subject_ids = data['subject']

    logger.info("Raw data shape: %s  (subjects: %s)", X_raw.shape, args.subjects)

    X_proc = preprocess_pipeline(X_raw, fs=data['fs'])

    # ── 2. Split (subject-level, no leakage) ─────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_split(
        X_proc, y, subject_ids, seed=args.seed
    )
    logger.info("Train: %d  Val: %d  Test: %d", len(X_train), len(X_val), len(X_test))

    # ── 3. Inject noise ───────────────────────────────────────────────────────
    noise_dict_train = inject_noise(X_train, snr_db_list=[args.snr])
    noise_dict_val   = inject_noise(X_val,   snr_db_list=[args.snr])

    X_train_noisy = noise_dict_train[f'noisy_{args.snr}']
    X_val_noisy   = noise_dict_val[f'noisy_{args.snr}']

    # ── 4. DataLoaders ────────────────────────────────────────────────────────
    train_loader = build_channel_dataset(X_train, X_train_noisy,
                                         batch_size=args.batch_size, shuffle=True)
    val_loader   = build_channel_dataset(X_val, X_val_noisy,
                                         batch_size=args.batch_size, shuffle=False)
    logger.info("Train batches: %d  Val batches: %d",
                len(train_loader), len(val_loader))

    # ── 5. Build model ────────────────────────────────────────────────────────
    cfg = DiffusionConfig(
        n_steps=args.n_steps,
        schedule=args.schedule,
        signal_length=750,
        in_channels=1,
        model_channels=32,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        dropout=0.0,
        time_emb_dim=128,
    )
    diffusion = GaussianDiffusion(cfg)
    model = UNet1D(cfg)
    logger.info("Model parameters: %d (%.1fK)",
                model.num_parameters, model.num_parameters / 1000)

    # ── 6. Train ──────────────────────────────────────────────────────────────
    t0 = time.time()
    history = train_diffusion(
        model, diffusion, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        early_stop_patience=args.early_stop,
        device='cpu',
    )
    elapsed = time.time() - t0
    logger.info("Training finished in %.1f minutes (%.1f hours)",
                elapsed / 60, elapsed / 3600)
    logger.info("Best epoch: %d  best_val_loss: %.5f",
                history['best_epoch'], history['best_val_loss'])

    # ── 7. Save loss curve ────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['train_losses'], label='Train loss', linewidth=1.2)
    ax.plot(history['val_losses'],   label='Val loss',   linewidth=1.2)
    ax.axvline(history['best_epoch'] - 1, color='red', linestyle='--',
               linewidth=0.8, label=f"Best epoch {history['best_epoch']}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Diffusion Teacher – Training Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/diffusion_training_curve.png', dpi=120)
    plt.close()
    logger.info("Loss curve saved to results/diffusion_training_curve.png")

    logger.info("Done. Run 'python src/verify_diffusion.py' to evaluate.")


if __name__ == '__main__':
    main()
