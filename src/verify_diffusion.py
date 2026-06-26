"""
verify_diffusion.py – Evaluate a trained DDPM teacher checkpoint.

Loads the best checkpoint from models/diffusion_teacher/best_model.pt, then:
  1. Runs denoising on a 50-trial test subset (subject 1) with 10, 25, 50 steps.
  2. Reports SNR improvement and MSE over raw noisy input.
  3. Benchmarks inference latency (100 repetitions, reports mean/std/p95).
  4. Saves results/diffusion_metrics.csv.
  5. Saves a comparison waveform plot.

If no checkpoint exists (training not yet run), the script falls back to an
UNTRAINED model and prints a clear warning - useful for smoke-testing the
inference pipeline before overnight training.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import load_bci_competition_data, inject_noise
from diffusion import DiffusionConfig, GaussianDiffusion, UNet1D
from metrics import compute_snr, compute_mse

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger('verify_diffusion')

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = os.path.join('models', 'diffusion_teacher', 'best_model.pt')
RESULTS_DIR     = 'results'
SUBJECT         = 1
N_TRIALS        = 50        # test trials
N_CHANNELS      = 22
SNR_INJECT_DB   = 10        # noise level used during training
STEP_COUNTS     = [10, 25, 50]
BENCH_REPS      = 30        # repetitions for latency benchmark (reduced for speed)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> tuple[UNet1D, GaussianDiffusion, bool]:
    """Load the model from checkpoint, or return untrained model with a warning."""
    cfg = DiffusionConfig()   # default config
    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Restore config from checkpoint if saved
        if 'cfg' in ckpt and ckpt['cfg'] is not None:
            cfg = ckpt['cfg']
        model = UNet1D(cfg)
        model.load_state_dict(ckpt['model_state'])
        logger.info("Loaded checkpoint from epoch %d (val_loss=%.5f)",
                    ckpt.get('epoch', -1), ckpt.get('val_loss', float('nan')))
        return model, GaussianDiffusion(cfg), False
    else:
        warnings.warn(
            f"No checkpoint found at '{checkpoint_path}'. "
            "Using UNTRAINED model – results are meaningless for SNR, "
            "but the inference pipeline will be fully smoke-tested.",
            UserWarning, stacklevel=2,
        )
        model = UNet1D(cfg)
        return model, GaussianDiffusion(cfg), True


def channel_snr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Average SNR across all trials and channels."""
    vals = []
    for t in range(clean.shape[0]):
        for c in range(clean.shape[1]):
            v = compute_snr(clean[t, c], denoised[t, c])
            if np.isfinite(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else float('nan')


def channel_mse(clean: np.ndarray, denoised: np.ndarray) -> float:
    return float(np.mean([
        compute_mse(clean[t, c], denoised[t, c])
        for t in range(clean.shape[0])
        for c in range(clean.shape[1])
    ]))


def run_denoising(
    model: UNet1D,
    diffusion: GaussianDiffusion,
    X_noisy: np.ndarray,
    steps: int,
) -> np.ndarray:
    """
    Denoise X_noisy using the diffusion model with `steps` reverse steps.

    Parameters
    ----------
    X_noisy : ndarray, shape (n_trials, n_channels, 750)

    Returns
    -------
    ndarray, shape (n_trials, n_channels, 750)
    """
    n_trials, n_ch, n_samp = X_noisy.shape
    model.eval()

    # Flatten channels → independent samples
    x_flat = torch.from_numpy(
        X_noisy.reshape(n_trials * n_ch, 1, n_samp).astype(np.float32)
    )

    with torch.no_grad():
        x_out = diffusion.denoise(model, x_flat, steps=steps)

    # Reshape back
    return x_out.numpy().reshape(n_trials, n_ch, n_samp)


def benchmark_latency(
    model: UNet1D,
    diffusion: GaussianDiffusion,
    steps: int,
    n_channels: int = 22,
    reps: int = BENCH_REPS,
) -> dict:
    """
    Measure per-window inference latency (single EEG trial, all channels).

    Each repetition denoises one trial (n_channels windows of 750 samples).
    """
    x_dummy = torch.randn(n_channels, 1, 750)
    model.eval()
    times = []
    torch.set_num_threads(1)   # single-thread CPU measurement

    # Warm-up
    with torch.no_grad():
        diffusion.denoise(model, x_dummy, steps=steps)

    for _ in range(reps):
        t0 = time.perf_counter()
        with torch.no_grad():
            diffusion.denoise(model, x_dummy, steps=steps)
        times.append((time.perf_counter() - t0) * 1000.0)   # ms

    torch.set_num_threads(torch.get_num_threads())  # restore
    times = np.array(times)
    return {
        'mean': float(np.mean(times)),
        'std':  float(np.std(times)),
        'p95':  float(np.percentile(times, 95)),
        'min':  float(np.min(times)),
        'max':  float(np.max(times)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model, diffusion, is_untrained = load_model(CHECKPOINT_PATH)
    logger.info("Model parameters: %d (%.1fK)",
                model.num_parameters, model.num_parameters / 1000)

    if is_untrained:
        logger.warning("*** Running with UNTRAINED model – SNR metrics not meaningful ***")

    # ── 2. Load test data ─────────────────────────────────────────────────────
    logger.info("Loading test data for subject %d …", SUBJECT)
    data = load_bci_competition_data(subjects=[SUBJECT])
    X_all  = data['X']                   # (288, 22, 750)
    X_clean = X_all[:N_TRIALS]           # (50, 22, 750)

    noisy_dict = inject_noise(X_clean, snr_db_list=[SNR_INJECT_DB])
    X_noisy = noisy_dict[f'noisy_{SNR_INJECT_DB}']   # (50, 22, 750)

    snr_noisy = channel_snr(X_clean, X_noisy)
    logger.info("Baseline (no denoising) SNR = %.2f dB", snr_noisy)

    # ── 3. Evaluate each step count ───────────────────────────────────────────
    rows = []
    denoised_results = {}

    for steps in STEP_COUNTS:
        logger.info("Denoising with %d steps …", steps)
        X_denoised = run_denoising(model, diffusion, X_noisy, steps)

        snr_val = channel_snr(X_clean, X_denoised)
        mse_val = channel_mse(X_clean, X_denoised)
        improvement = snr_val - snr_noisy

        # Latency benchmark
        lat = benchmark_latency(model, diffusion, steps, n_channels=N_CHANNELS,
                                reps=BENCH_REPS)

        logger.info(
            "steps=%2d  SNR=%7.2f dB  dSNR=%+.2f dB  MSE=%.4f  "
            "latency: mean=%.1f ms  std=%.1f ms  p95=%.1f ms",
            steps, snr_val, improvement, mse_val,
            lat['mean'], lat['std'], lat['p95'],
        )
        rows.append({
            'steps':        steps,
            'snr_db':       round(snr_val, 4),
            'snr_improvement_db': round(improvement, 4),
            'mse':          round(mse_val, 6),
            'latency_mean': round(lat['mean'], 2),
            'latency_std':  round(lat['std'],  2),
            'latency_p95':  round(lat['p95'],  2),
            'latency_min':  round(lat['min'],  2),
            'latency_max':  round(lat['max'],  2),
            'is_untrained': is_untrained,
        })
        denoised_results[steps] = X_denoised

    # ── 4. Save CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, 'diffusion_metrics.csv')
    fieldnames = ['steps', 'snr_db', 'snr_improvement_db', 'mse',
                  'latency_mean', 'latency_std', 'latency_p95',
                  'latency_min', 'latency_max', 'is_untrained']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Metrics saved to %s", csv_path)

    # ── 5. Waveform comparison plot ───────────────────────────────────────────
    t_axis = np.arange(750) / 250.0   # seconds
    trial, ch = 0, 0                   # plot first trial, first channel

    n_plots = 2 + len(STEP_COUNTS)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.8 * n_plots))
    fig.suptitle(f'Diffusion Denoising – Trial 0, Channel 0'
                 + (' [UNTRAINED]' if is_untrained else ''), fontsize=12)

    axes[0].plot(t_axis, X_clean[trial, ch], lw=0.8, color='steelblue')
    axes[0].set_title('Clean (ground truth)')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(t_axis, X_noisy[trial, ch], lw=0.8, color='tomato')
    axes[1].set_title(f'Noisy (injected @ {SNR_INJECT_DB} dB, SNR={snr_noisy:.1f} dB)')
    axes[1].set_ylabel('Amplitude')

    colors = ['#2ecc71', '#27ae60', '#1e8449']
    for i, steps in enumerate(STEP_COUNTS):
        ax = axes[i + 2]
        row = next(r for r in rows if r['steps'] == steps)
        ax.plot(t_axis, denoised_results[steps][trial, ch],
                lw=0.8, color=colors[i])
        ax.set_title(
            f'Diffusion {steps} steps  '
            f'(dSNR={row["snr_improvement_db"]:+.2f} dB, '
            f'latency p95={row["latency_p95"]:.0f} ms)'
        )
        ax.set_ylabel('Amplitude')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'diffusion_comparison.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()
    logger.info("Comparison plot saved to %s", plot_path)

    # ── 6. Parameter count summary ────────────────────────────────────────────
    n_params = model.num_parameters
    print(f"\n{'='*55}")
    print(f"  Model parameters : {n_params:,}  ({n_params/1e3:.1f}K)")
    print(f"  Constraint check : {'PASS' if n_params < 500_000 else 'FAIL'}"
          f"  (< 500K required)")
    print(f"\n  Step-count results:")
    for row in rows:
        print(f"    steps={row['steps']:2d}  "
              f"SNR={row['snr_db']:6.2f} dB  "
              f"dSNR={row['snr_improvement_db']:+.2f} dB  "
              f"p95_latency={row['latency_p95']:.0f} ms")
    if is_untrained:
        print("\n  [WARNING] Model is UNTRAINED. Run train_diffusion.py first.")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
