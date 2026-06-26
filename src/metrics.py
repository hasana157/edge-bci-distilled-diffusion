"""
metrics.py – Evaluation metrics for EEG denoising.

All functions operate on numpy arrays and follow NumPy-style docstrings.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_snr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in dB.

    Parameters
    ----------
    clean : np.ndarray
        Ground-truth clean signal (any shape).
    denoised : np.ndarray
        Reconstructed / denoised signal (same shape as clean).

    Returns
    -------
    float
        SNR in dB. Returns -inf if the residual is zero (perfect reconstruction)
        and returns np.nan if the clean signal has zero power.
    """
    clean = np.asarray(clean, dtype=np.float64)
    denoised = np.asarray(denoised, dtype=np.float64)

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - denoised) ** 2)

    if signal_power == 0.0:
        logger.warning("compute_snr: clean signal has zero power. Returning nan.")
        return np.nan
    if noise_power == 0.0:
        return np.inf

    return 10.0 * np.log10(signal_power / noise_power)


def compute_mse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE) between clean and denoised signals.

    Parameters
    ----------
    clean : np.ndarray
        Ground-truth clean signal.
    denoised : np.ndarray
        Denoised signal.

    Returns
    -------
    float
        MSE value (non-negative).
    """
    return float(np.mean((np.asarray(clean, dtype=np.float64) -
                          np.asarray(denoised, dtype=np.float64)) ** 2))


def compute_psnr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB.

    Parameters
    ----------
    clean : np.ndarray
        Ground-truth clean signal.
    denoised : np.ndarray
        Denoised signal.

    Returns
    -------
    float
        PSNR in dB.
    """
    clean = np.asarray(clean, dtype=np.float64)
    mse = compute_mse(clean, denoised)
    if mse == 0.0:
        return np.inf
    peak = np.max(np.abs(clean))
    if peak == 0.0:
        return np.nan
    return 20.0 * np.log10(peak) - 10.0 * np.log10(mse)


def compute_all_metrics(
    clean: np.ndarray,
    denoised: np.ndarray,
    prefix: str = ''
) -> dict:
    """
    Compute all available evaluation metrics in a single call.

    Parameters
    ----------
    clean : np.ndarray
        Ground-truth clean signal.
    denoised : np.ndarray
        Denoised signal.
    prefix : str, optional
        String prefix applied to every key in the returned dict.

    Returns
    -------
    dict
        Dictionary with keys ``snr_db``, ``mse``, ``psnr`` (each optionally
        prefixed by ``prefix``).
    """
    results = {
        'snr_db': compute_snr(clean, denoised),
        'mse': compute_mse(clean, denoised),
        'psnr': compute_psnr(clean, denoised),
    }
    if prefix:
        results = {f"{prefix}{k}": v for k, v in results.items()}
    return results
