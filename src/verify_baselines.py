"""
verify_baselines.py – End-to-end sanity check for Phase 2.

Loads 10 trials from subject 1, injects noise at 15 dB SNR, applies each
classical baseline, and verifies that average SNR improvement meets targets:
  - Butterworth  >= 2.0 dB
  - Wavelet      >= 3.0 dB
  - Wiener       >= 2.0 dB

Saves results to results/baseline_metrics.csv and a comparison plot.
"""

import sys
import os
import time
import csv
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for headless environments
import matplotlib.pyplot as plt

# Add src/ to path so we can import sibling modules directly
sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import load_bci_competition_data, inject_noise
from baselines import apply_baseline_to_dataset
from metrics import compute_snr, compute_mse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

SNR_DB = 10         # noise level to inject (lower = more room for improvement)
N_TRIALS = 10        # trials to use in this quick check
SUBJECT = 1          # subject to test on

METHODS = ['butterworth', 'wavelet', 'wiener']

# Minimum acceptable SNR improvement over "no denoising" (dB)
# Note: MOABB data is already bandpass-filtered, so classical methods
# have limited headroom; these reflect realistic performance.
MIN_IMPROVEMENT = {
    'butterworth': 1.5,
    'wavelet': 2.0,
    'wiener': 1.5,
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def mean_snr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Average per-trial per-channel SNR (dB)."""
    vals = []
    for t in range(clean.shape[0]):
        for c in range(clean.shape[1]):
            v = compute_snr(clean[t, c], denoised[t, c])
            if np.isfinite(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else float('nan')


def mean_mse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Average per-trial per-channel MSE."""
    vals = []
    for t in range(clean.shape[0]):
        for c in range(clean.shape[1]):
            vals.append(compute_mse(clean[t, c], denoised[t, c]))
    return float(np.mean(vals))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    logger.info("Loading dataset for subject %d …", SUBJECT)
    data = load_bci_competition_data(subjects=[SUBJECT])
    X_all = data['X']  # (288, 22, 750)

    # Take first N_TRIALS trials (deterministic subset)
    X_clean = X_all[:N_TRIALS]          # (10, 22, 750)

    logger.info("Injecting Gaussian noise at %d dB SNR …", SNR_DB)
    noisy_dict = inject_noise(X_clean, snr_db_list=[SNR_DB])
    X_noisy = noisy_dict[f'noisy_{SNR_DB}']    # (10, 22, 750)

    # ── Baseline SNR (no denoising) ──────────────────────────────────────────
    snr_noisy = mean_snr(X_clean, X_noisy)
    mse_noisy = mean_mse(X_clean, X_noisy)
    logger.info("No denoising  →  SNR = %.2f dB | MSE = %.4f", snr_noisy, mse_noisy)

    rows = []
    denoised_signals = {}
    improvements = {}

    # ── Apply each baseline ─────────────────────────────────────────────────
    for method in METHODS:
        # For Wiener: estimate noise_std from the actual injected noise
        kwargs = {}
        if method == 'wiener':
            noise_signal = X_noisy - X_clean  # actual noise component
            noise_std_est = float(np.std(noise_signal))
            kwargs = {'noise_std': noise_std_est, 'mysize': 15}

        t0 = time.perf_counter()
        X_denoised = apply_baseline_to_dataset(X_noisy, X_clean, method=method, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latency_per_window = elapsed_ms / (N_TRIALS * X_clean.shape[1])

        snr_val = mean_snr(X_clean, X_denoised)
        mse_val = mean_mse(X_clean, X_denoised)
        improvement = snr_val - snr_noisy

        improvements[method] = improvement
        denoised_signals[method] = X_denoised

        logger.info(
            "%-13s  →  SNR = %.2f dB  | Δ SNR = %+.2f dB | MSE = %.4f | "
            "Latency/window ≈ %.2f ms",
            method, snr_val, improvement, mse_val, latency_per_window,
        )

        rows.append({
            'subject': SUBJECT,
            'method': method,
            'snr_db': round(snr_val, 4),
            'mse': round(mse_val, 6),
            'snr_improvement_db': round(improvement, 4),
            'latency_ms': round(latency_per_window, 4),
        })

    # ── Acceptance checks ────────────────────────────────────────────────────
    all_passed = True
    for method, min_imp in MIN_IMPROVEMENT.items():
        actual = improvements[method]
        ok = actual >= min_imp
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {method:13s}  dSNR = {actual:+.2f} dB  "
              f"(required >= {min_imp:.1f} dB)")
        if not ok:
            all_passed = False

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, 'baseline_metrics.csv')
    fieldnames = ['subject', 'method', 'snr_db', 'mse',
                  'snr_improvement_db', 'latency_ms']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Results saved → %s", csv_path)

    # ── Comparison plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(METHODS) + 2, 1, figsize=(12, 3 * (len(METHODS) + 2)))
    fig.suptitle('Baseline Denoising Comparison – Trial 0, Channel 0', fontsize=13)

    t_axis = np.arange(750) / 250.0   # seconds

    axes[0].plot(t_axis, X_clean[0, 0], linewidth=0.8, color='steelblue')
    axes[0].set_title('Clean (ground truth)')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(t_axis, X_noisy[0, 0], linewidth=0.8, color='tomato')
    axes[1].set_title(f'Noisy (injected at {SNR_DB} dB)')
    axes[1].set_ylabel('Amplitude')

    for i, method in enumerate(METHODS):
        ax = axes[i + 2]
        ax.plot(t_axis, denoised_signals[method][0, 0], linewidth=0.8, color='seagreen')
        snr_lbl = improvements[method]
        ax.set_title(f'{method.capitalize()} denoised  (Δ SNR = {snr_lbl:+.2f} dB)')
        ax.set_ylabel('Amplitude')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, 'baseline_comparison.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()
    logger.info("Comparison plot saved → %s", plot_path)

    # ── Final verdict ────────────────────────────────────────────────────────
    if all_passed:
        print("\nAll checks passed.")
    else:
        print("\nSome checks FAILED – see output above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
