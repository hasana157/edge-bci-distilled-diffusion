"""
metrics.py – Evaluation metric helpers for EEG denoising and classification.

Covers:
  - SNR / SNR improvement
  - MSE / RMSE
  - Pearson correlation coefficient
  - Classification accuracy, per-class accuracy
  - Kappa coefficient (BCI standard metric)
  - Throughput and latency statistics summary
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio in dB.

    SNR = 10 * log10(E[signal²] / E[noise²])

    Parameters
    ----------
    signal : reference clean signal (any shape)
    noise  : noise component (same shape)

    Returns
    -------
    float – SNR in dB
    """
    sp = np.mean(signal**2) + 1e-10
    np_ = np.mean(noise**2) + 1e-10
    return float(10.0 * np.log10(sp / np_))


def snr_improvement(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
) -> float:
    """
    SNR improvement = SNR_output − SNR_input (dB).

    Positive values indicate the denoiser improved signal quality.
    Target: 5–8 dB (SRS UPGRADE from 3–5 dB).
    """
    snr_in = snr_db(clean, noisy - clean)
    snr_out = snr_db(clean, denoised - clean)
    return float(snr_out - snr_in)


def mse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((clean - denoised) ** 2))


def rmse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((clean - denoised) ** 2)))


def pearson_correlation(clean: np.ndarray, denoised: np.ndarray) -> float:
    """
    Average Pearson correlation coefficient between clean and denoised signals.

    Flattens both arrays before computing to get a single scalar.
    """
    c = clean.flatten()
    d = denoised.flatten()
    if np.std(c) < 1e-10 or np.std(d) < 1e-10:
        return 0.0
    return float(np.corrcoef(c, d)[0, 1])


def compute_all_signal_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
) -> Dict[str, float]:
    """Compute SNR improvement, MSE, RMSE, and Pearson correlation in one call."""
    return {
        "snr_improvement_db": snr_improvement(clean, noisy, denoised),
        "mse": mse(clean, denoised),
        "rmse": rmse(clean, denoised),
        "pearson_correlation": pearson_correlation(clean, denoised),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall classification accuracy (0–1)."""
    return float(np.mean(y_true == y_pred))


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 4,
) -> np.ndarray:
    """Per-class accuracy array of length n_classes."""
    accs = np.zeros(n_classes)
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            accs[c] = np.mean(y_pred[mask] == c)
    return accs


def cohen_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 4,
) -> float:
    """
    Cohen's Kappa coefficient – standard BCI performance metric.

    κ = (p_o − p_e) / (1 − p_e)
    where p_o = observed agreement, p_e = expected agreement by chance.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0

    conf_mat = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            conf_mat[t, p] += 1

    p_o = np.trace(conf_mat) / n
    row_sums = conf_mat.sum(axis=1)
    col_sums = conf_mat.sum(axis=0)
    p_e = np.sum(row_sums * col_sums) / (n**2)
    if abs(1 - p_e) < 1e-10:
        return 1.0
    return float((p_o - p_e) / (1 - p_e))


def compute_all_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 4,
) -> Dict[str, float]:
    """Compute accuracy, per-class accuracy, and kappa in one call."""
    result: Dict[str, float] = {
        "accuracy": accuracy(y_true, y_pred),
        "kappa": cohen_kappa(y_true, y_pred, n_classes),
    }
    pca = per_class_accuracy(y_true, y_pred, n_classes)
    class_names = ["left", "right", "feet", "tongue"]
    for i, name in enumerate(class_names[:n_classes]):
        result[f"acc_{name}"] = float(pca[i])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Latency statistics summary
# ─────────────────────────────────────────────────────────────────────────────

def latency_stats(times_ms: List[float]) -> Dict[str, float]:
    """
    Compute standard latency statistics from a list of measurements.

    Parameters
    ----------
    times_ms : list of float – latencies in milliseconds

    Returns
    -------
    dict with keys: mean, std, min, max, p50, p95, p99 (all in ms)
    """
    arr = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def print_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print a table of metrics for multiple models."""
    if not metrics_dict:
        return

    # Determine all metric keys
    all_keys = []
    for v in metrics_dict.values():
        for k in v:
            if k not in all_keys:
                all_keys.append(k)

    col_w = 16
    header = f"{'Model':<25}" + "".join(f"{k:>{col_w}}" for k in all_keys)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for model_name, vals in metrics_dict.items():
        row = f"{model_name:<25}"
        for k in all_keys:
            v = vals.get(k, float("nan"))
            row += f"{v:>{col_w}.4f}"
        print(row)
    print("=" * len(header) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible aliases (used by verify_diffusion.py, verify_baselines.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """
    Compute SNR of the denoised signal relative to the clean reference.
    SNR = 10 * log10(E[clean²] / E[(clean - denoised)²])
    """
    return snr_db(clean, clean - denoised)


def compute_mse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Alias for mse() – backward-compatible name."""
    return mse(clean, denoised)
