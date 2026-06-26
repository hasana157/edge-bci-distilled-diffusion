"""
baselines.py – Classical EEG denoising baselines.

FR-201  Butterworth bandpass filter (4–40 Hz, order 4).
FR-202  Wavelet denoising (Daubechies-4, BayesShrink).
FR-203  Wiener filtering.
FR-204  SNR metric helper.

All functions operate on NumPy arrays for speed on CPU.
They are also callable from PyTorch tensors via the thin wrappers at the bottom.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FR-201  Butterworth bandpass
# ─────────────────────────────────────────────────────────────────────────────

def butterworth_bandpass(
    eeg: np.ndarray,
    lowcut: float = 4.0,
    highcut: float = 40.0,
    fs: float = 250.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    eeg     : np.ndarray – shape (..., n_samples)
    lowcut  : low cutoff frequency in Hz
    highcut : high cutoff frequency in Hz
    fs      : sampling rate in Hz
    order   : filter order (≥4 per FR-201)

    Returns
    -------
    np.ndarray – filtered signal, same shape as input
    """
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, eeg, axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FR-202  Wavelet denoising (BayesShrink)
# ─────────────────────────────────────────────────────────────────────────────

def _bayesshrink_threshold(coeffs: np.ndarray) -> float:
    """
    BayesShrink soft threshold for a wavelet coefficient array.

    σ_n  = median(|d|) / 0.6745  (robust noise estimate)
    σ_x² = max(0, std(d)² − σ_n²)
    T    = σ_n² / σ_x   (or ∞ if σ_x = 0 → zero the band)
    """
    sigma_n = np.median(np.abs(coeffs)) / 0.6745
    sigma_x2 = max(0.0, np.var(coeffs) - sigma_n**2)
    if sigma_x2 == 0.0:
        return np.inf
    return (sigma_n**2) / np.sqrt(sigma_x2)


def wavelet_denoise(
    eeg: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> np.ndarray:
    """
    Denoise EEG using DWT soft-thresholding with BayesShrink.

    Parameters
    ----------
    eeg     : np.ndarray – shape (..., n_samples)
    wavelet : PyWavelets wavelet name (default: Daubechies-4)
    level   : decomposition level

    Returns
    -------
    np.ndarray – denoised signal, same shape as input
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("PyWavelets is required. pip install PyWavelets")

    original_shape = eeg.shape
    flat = eeg.reshape(-1, eeg.shape[-1])
    out = np.empty_like(flat)

    for i in range(flat.shape[0]):
        coeffs = pywt.wavedec(flat[i], wavelet, level=level)
        thresholded = [coeffs[0]]  # keep approximation unchanged
        for detail in coeffs[1:]:
            thr = _bayesshrink_threshold(detail)
            thresholded.append(pywt.threshold(detail, thr, mode="soft"))
        out[i] = pywt.waverec(thresholded, wavelet)[: flat.shape[-1]]

    return out.reshape(original_shape).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FR-203  Wiener filter
# ─────────────────────────────────────────────────────────────────────────────

def wiener_filter(
    eeg: np.ndarray,
    noise_psd: Optional[np.ndarray] = None,
    fs: float = 250.0,
    n_fft: Optional[int] = None,
) -> np.ndarray:
    """
    Frequency-domain Wiener filter for EEG denoising.

    If ``noise_psd`` is not provided, the noise PSD is estimated from
    the median of the signal PSD (assumes noise dominates at high freq).

    Parameters
    ----------
    eeg       : np.ndarray – shape (..., n_samples)
    noise_psd : optional 1-D noise PSD estimate (length n_fft//2+1)
    fs        : sampling rate
    n_fft     : FFT size (defaults to signal length)

    Returns
    -------
    np.ndarray – denoised signal, same shape as input
    """
    original_shape = eeg.shape
    flat = eeg.reshape(-1, eeg.shape[-1])
    L = flat.shape[-1]
    nfft = n_fft or L
    out = np.empty_like(flat)

    for i in range(flat.shape[0]):
        X = np.fft.rfft(flat[i], n=nfft)
        Pxx = np.abs(X) ** 2  # signal PSD estimate

        if noise_psd is None:
            # Estimate noise floor as minimum of smoothed PSD
            Pnn = np.full_like(Pxx, np.percentile(Pxx, 10))
        else:
            Pnn = noise_psd[: len(Pxx)]

        # Wiener gain: H(f) = (Pxx - Pnn) / Pxx, clamped to [0, 1]
        H = np.clip((Pxx - Pnn) / (Pxx + 1e-10), 0.0, 1.0)
        out[i] = np.fft.irfft(H * X, n=nfft)[:L]

    return out.reshape(original_shape).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr_improvement(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
) -> float:
    """
    Compute SNR improvement in dB.

    SNR_improvement = SNR_output − SNR_input

    Parameters
    ----------
    clean    : reference clean signal
    noisy    : noisy input
    denoised : model/filter output

    Returns
    -------
    float – SNR improvement in dB (positive = better)
    """
    def _snr(signal, noise):
        signal_power = np.mean(signal**2) + 1e-10
        noise_power = np.mean(noise**2) + 1e-10
        return 10.0 * np.log10(signal_power / noise_power)

    noise_in = noisy - clean
    noise_out = denoised - clean
    snr_in = _snr(clean, noise_in)
    snr_out = _snr(clean, noise_out)
    return float(snr_out - snr_in)


def compute_mse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Mean Squared Error between clean and denoised signals."""
    return float(np.mean((clean - denoised) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run all baselines on a batch and return dict of results
# ─────────────────────────────────────────────────────────────────────────────

def run_all_baselines(
    clean: np.ndarray,
    noisy: np.ndarray,
    fs: float = 250.0,
) -> dict:
    """
    Apply Butterworth, Wavelet, and Wiener filters; return SNR and MSE dict.

    Parameters
    ----------
    clean : np.ndarray – shape (n_trials, n_channels, n_samples)
    noisy : np.ndarray – same shape as clean

    Returns
    -------
    dict with keys 'butterworth', 'wavelet', 'wiener' each mapping to
    {'snr_improvement_db': float, 'mse': float, 'denoised': np.ndarray}
    """
    results = {}

    for name, fn in [
        ("butterworth", lambda x: butterworth_bandpass(x, fs=fs)),
        ("wavelet", wavelet_denoise),
        ("wiener", wiener_filter),
    ]:
        try:
            denoised = fn(noisy)
            snr_imp = compute_snr_improvement(clean, noisy, denoised)
            mse = compute_mse(clean, denoised)
            results[name] = {
                "snr_improvement_db": snr_imp,
                "mse": mse,
                "denoised": denoised,
            }
            logger.info(
                "Baseline %-12s  SNR_imp=%.2f dB  MSE=%.5f",
                name, snr_imp, mse,
            )
        except Exception as exc:
            logger.warning("Baseline %s failed: %s", name, exc)
            results[name] = {"snr_improvement_db": 0.0, "mse": float("inf"), "denoised": noisy}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible helper used by verify_baselines.py
# ─────────────────────────────────────────────────────────────────────────────

def apply_baseline_to_dataset(
    noisy: np.ndarray,
    clean: np.ndarray,
    method: str = "butterworth",
    fs: float = 250.0,
    **kwargs,
) -> np.ndarray:
    """
    Apply a named denoising baseline to an entire dataset array.

    Parameters
    ----------
    noisy  : np.ndarray – noisy EEG, shape (n_trials, n_channels, n_samples)
    clean  : np.ndarray – clean EEG (used only for Wiener noise estimation)
    method : 'butterworth' | 'wavelet' | 'wiener'
    fs     : sampling rate in Hz (for Butterworth)
    **kwargs : passed to the underlying filter (e.g. noise_std for wiener)

    Returns
    -------
    np.ndarray – denoised EEG, same shape as noisy
    """
    if method == "butterworth":
        return butterworth_bandpass(noisy, fs=fs)
    elif method == "wavelet":
        return wavelet_denoise(noisy)
    elif method == "wiener":
        # Ignore noise_std / mysize kwargs – use our frequency-domain Wiener
        return wiener_filter(noisy, fs=fs)
    else:
        raise ValueError(f"Unknown baseline method: {method!r}. "
                         f"Choose from 'butterworth', 'wavelet', 'wiener'.")
