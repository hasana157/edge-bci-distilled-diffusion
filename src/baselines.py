"""
baselines.py – Classical EEG denoising baselines.

Methods included:
  - Butterworth bandpass filter
  - Wavelet denoising (BayesShrink thresholding)
  - Wiener filter (local or frequency-domain)

All functions operate channel-independently on 1-D numpy arrays and are
designed for CPU-only execution.  NumPy-style docstrings are used throughout.
"""

import numpy as np
import scipy.signal
import pywt
import logging
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _validate_1d(signal: np.ndarray, name: str = "signal") -> np.ndarray:
    """Return a clean 1-D float64 array, raising ValueError on bad input."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        logger.warning(
            "%s contains NaN/Inf values; replacing with zeros.", name
        )
        arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


# ---------------------------------------------------------------------------
# 1. Butterworth bandpass filter
# ---------------------------------------------------------------------------

def butterworth_denoise(
    signal: np.ndarray,
    fs: float = 250.0,
    lowcut: float = 4.0,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to a 1-D EEG channel.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Raw or noisy 1-D EEG signal.
    fs : float, default=250.0
        Sampling frequency in Hz.
    lowcut : float, default=4.0
        Lower -3 dB cutoff frequency in Hz (must be > 0).
    highcut : float, default=40.0
        Upper -3 dB cutoff frequency in Hz (must be < fs / 2).
    order : int, default=4
        Filter order (must be >= 1).

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Bandpass-filtered signal of the same length as the input.

    Notes
    -----
    Uses ``scipy.signal.butter`` + ``filtfilt`` for zero-phase (forward-
    backward) filtering, eliminating phase distortion.

    Examples
    --------
    >>> import numpy as np
    >>> sig = np.random.randn(750)
    >>> out = butterworth_denoise(sig, fs=250, lowcut=4, highcut=40)
    >>> out.shape
    (750,)
    """
    arr = _validate_1d(signal)
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='bandpass')
    return scipy.signal.filtfilt(b, a, arr)


# ---------------------------------------------------------------------------
# 2. Wavelet denoising (BayesShrink)
# ---------------------------------------------------------------------------

def _bayesshrink_threshold(detail_coeffs: np.ndarray, sigma_n: float) -> float:
    """
    Compute BayesShrink soft threshold for one wavelet detail subband.

    Noise standard deviation ``sigma_n`` is estimated **once** from the
    finest detail level (highest frequency subband) using the MAD estimator
    and passed in here for every other level::

        sigma_n = median(|d_finest|) / 0.6745

    Per-level signal variance is then estimated as::

        sigma_s^2 = max(0, mean(d^2) - sigma_n^2)

    The threshold is::

        T = sigma_n^2 / sigma_s   (or +inf when sigma_s == 0)

    Parameters
    ----------
    detail_coeffs : np.ndarray
        1-D array of wavelet detail coefficients at this decomposition level.
    sigma_n : float
        Noise standard deviation estimated from the finest detail level.

    Returns
    -------
    float
        Soft threshold value.
    """
    sigma_n2 = sigma_n ** 2
    sigma_s2 = max(0.0, np.mean(detail_coeffs ** 2) - sigma_n2)
    if sigma_s2 == 0.0:
        return np.inf
    return sigma_n2 / np.sqrt(sigma_s2)


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: int = 4,
    method: str = 'BayesShrink',
) -> np.ndarray:
    """
    Denoise a 1-D EEG signal using wavelet thresholding.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Raw or noisy 1-D EEG signal.
    wavelet : str, default='db4'
        Wavelet family/name supported by ``pywt`` (e.g., ``'db4'``, ``'sym4'``).
    level : int, default=4
        Number of decomposition levels.  If the signal is too short for the
        requested level, the level is automatically reduced.
    method : str, default='BayesShrink'
        Thresholding method.  Currently only ``'BayesShrink'`` is supported;
        falls back to a universal threshold otherwise.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Denoised signal of the same length as the input.

    Notes
    -----
    Soft thresholding is applied to all detail coefficients.  The approximation
    coefficients (lowest frequency band) are kept unchanged to preserve signal
    energy.

    Examples
    --------
    >>> import numpy as np
    >>> sig = np.random.randn(750)
    >>> out = wavelet_denoise(sig, wavelet='db4', level=4)
    >>> out.shape
    (750,)
    """
    arr = _validate_1d(signal)

    # Clamp level to maximum supported by the signal length
    max_level = pywt.dwt_max_level(len(arr), wavelet)
    level = min(level, max_level)

    # Decompose
    coeffs = pywt.wavedec(arr, wavelet, level=level)
    # coeffs[0] = approximation, coeffs[1..level] = details (finest = coeffs[-1])

    # Estimate noise ONCE from the finest detail level (highest-frequency subband)
    finest_detail = coeffs[-1]
    sigma_n = np.median(np.abs(finest_detail)) / 0.6745

    # Threshold each detail level using the shared sigma_n
    new_coeffs = [coeffs[0]]  # keep approximation coefficients unchanged
    for detail in coeffs[1:]:
        if method == 'BayesShrink':
            thresh = _bayesshrink_threshold(detail, sigma_n)
        else:
            # Universal (VisuShrink) fallback
            thresh = sigma_n * np.sqrt(2.0 * np.log(len(arr)))
        # Soft thresholding
        thresholded = pywt.threshold(detail, thresh, mode='soft')
        new_coeffs.append(thresholded)

    # Reconstruct
    denoised = pywt.waverec(new_coeffs, wavelet)

    # Align length (waverec can add one sample for odd-length signals)
    return denoised[:len(arr)]


# ---------------------------------------------------------------------------
# 3. Wiener filter
# ---------------------------------------------------------------------------

def wiener_denoise(
    signal: np.ndarray,
    noise_std: float = 1.0,
    mysize: int = 5,
) -> np.ndarray:
    """
    Apply a frequency-domain Wiener filter to a 1-D EEG signal.

    For white Gaussian noise with known standard deviation ``noise_std``, the
    optimal (MMSE) linear filter in the frequency domain has the gain::

        H(k) = S_xx(k) / (S_xx(k) + S_nn)

    where ``S_xx(k) = max(0, |X(k)|^2 - S_nn)`` is the estimated signal PSD
    (subtraction principle) and ``S_nn = noise_std^2 * N`` is the flat noise
    PSD per DFT bin for a signal of length ``N``.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Raw or noisy 1-D EEG signal.
    noise_std : float, default=1.0
        Estimated noise standard deviation.  Set to ``0`` to auto-estimate
        from the signal via the MAD wavelet estimator.
    mysize : int, default=5
        Unused (kept for API compatibility with the original local Wiener
        interface).  Will be removed in a future version.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Wiener-filtered signal of the same length as the input.

    Notes
    -----
    The spectral subtraction / Wiener approach is optimal for additive white
    Gaussian noise and outperforms the local (time-domain) Wiener filter for
    EEG because EEG power is concentrated in narrow frequency bands while
    noise power is spread uniformly across all frequencies.

    Examples
    --------
    >>> import numpy as np
    >>> sig = np.random.randn(750)
    >>> out = wiener_denoise(sig, noise_std=1.0)
    >>> out.shape
    (750,)
    """
    arr = _validate_1d(signal)
    N = len(arr)

    if noise_std <= 0.0:
        # Auto-estimate noise std from signal via MAD on finest wavelet detail
        import pywt as _pywt
        c = _pywt.wavedec(arr, 'db1', level=1)
        noise_std = max(np.median(np.abs(c[-1])) / 0.6745, 1e-10)

    # Noise PSD per DFT bin (flat spectrum for white noise)
    # E[|N(k)|^2] = noise_std^2 * N  for each bin k
    noise_psd = noise_std ** 2 * N

    # Forward DFT
    X = np.fft.rfft(arr)
    psd_x = np.abs(X) ** 2   # raw periodogram (signal + noise)

    # Estimate signal PSD via spectral subtraction (floored at 0)
    signal_psd = np.maximum(0.0, psd_x - noise_psd)

    # Wiener gain per frequency bin
    H = signal_psd / (signal_psd + noise_psd + 1e-30)

    # Apply gain and inverse DFT
    return np.fft.irfft(X * H, n=N)


# ---------------------------------------------------------------------------
# 4. Dataset-level wrapper
# ---------------------------------------------------------------------------

def apply_baseline_to_dataset(
    X_noisy: np.ndarray,
    X_clean: np.ndarray,
    method: str = 'butterworth',
    **kwargs,
) -> np.ndarray:
    """
    Apply a classical denoising baseline to every channel of every trial.

    Parameters
    ----------
    X_noisy : np.ndarray, shape (n_trials, n_channels, n_samples)
        Noisy EEG dataset.
    X_clean : np.ndarray, shape (n_trials, n_channels, n_samples)
        Corresponding ground-truth clean EEG dataset (used only for shape
        validation; the function does *not* use clean data during denoising).
    method : str, default='butterworth'
        One of ``'butterworth'``, ``'wavelet'``, ``'wiener'``.
    **kwargs
        Extra keyword arguments forwarded to the selected denoising function.

    Returns
    -------
    np.ndarray, shape (n_trials, n_channels, n_samples)
        Denoised dataset with the same shape as ``X_noisy``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported options.
    """
    _method_map: dict[str, Callable] = {
        'butterworth': butterworth_denoise,
        'wavelet': wavelet_denoise,
        'wiener': wiener_denoise,
    }
    if method not in _method_map:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {list(_method_map)}."
        )
    denoise_fn = _method_map[method]

    if X_noisy.shape != X_clean.shape:
        raise ValueError(
            f"X_noisy shape {X_noisy.shape} != X_clean shape {X_clean.shape}"
        )

    n_trials, n_channels, n_samples = X_noisy.shape
    X_denoised = np.zeros_like(X_noisy)

    for t in range(n_trials):
        for c in range(n_channels):
            X_denoised[t, c] = denoise_fn(X_noisy[t, c], **kwargs)

    return X_denoised
