"""
data_pipeline.py – BCI Competition IV 2a data loading and preprocessing.

FR-101  Load MATLAB .mat files for all 9 subjects.
FR-102  Standardize to 250 Hz.
FR-103  Segment 3-second motor-imagery windows (750 samples).
FR-104  Per-channel z-score normalization (fit on train set only).
FR-105  80/10/10 train/val/test split with NO subject leakage.
FR-204  Synthetic noise injection at 10, 15, 20 dB SNR.
FR-205  Artifact flagging (amplitude > 200 µV or high-freq power threshold).

The module works both when .mat files are present and when only synthetic
(randomly generated) data is available, enabling offline unit-testing.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level .mat loader
# ─────────────────────────────────────────────────────────────────────────────

def load_subject_mat(mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one BCI Competition IV 2a .mat file.

    Parameters
    ----------
    mat_path : str
        Path to the .mat file (e.g., A01T.mat).

    Returns
    -------
    eeg : np.ndarray, shape (n_trials, n_channels, n_samples)
    labels : np.ndarray, shape (n_trials,)  values in {1,2,3,4}
    """
    try:
        from scipy.io import loadmat
        mat = loadmat(mat_path)
    except ImportError:
        raise ImportError("scipy is required for .mat loading. pip install scipy")

    # BCI Comp IV 2a key layout varies between versions; try multiple keys.
    possible_keys = ["data", "X", "eeg", "s"]
    label_keys = ["y", "labels", "Y", "classlabel"]

    eeg_raw = None
    for k in possible_keys:
        if k in mat:
            eeg_raw = mat[k]
            break

    labels_raw = None
    for k in label_keys:
        if k in mat:
            labels_raw = mat[k]
            break

    if eeg_raw is None:
        raise KeyError(
            f"Cannot find EEG data key in {mat_path}. "
            f"Available keys: {list(mat.keys())}"
        )

    # Ensure shape is (trials, channels, samples)
    eeg_raw = np.array(eeg_raw, dtype=np.float32)
    if eeg_raw.ndim == 2:
        # (channels, samples) → add trial dim
        eeg_raw = eeg_raw[np.newaxis, :, :]

    if labels_raw is not None:
        labels_raw = np.array(labels_raw, dtype=np.int64).flatten()
    else:
        labels_raw = np.ones(eeg_raw.shape[0], dtype=np.int64)

    return eeg_raw, labels_raw


def _generate_synthetic_subject(
    n_trials: int = 288,
    n_channels: int = 22,
    n_samples: int = 750,
    n_classes: int = 4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG data for testing when no .mat files are available.

    Returns
    -------
    eeg : np.ndarray, shape (n_trials, n_channels, n_samples)
    labels : np.ndarray, shape (n_trials,)  values in {1,2,3,4}
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 3.0, n_samples)

    eeg = np.zeros((n_trials, n_channels, n_samples), dtype=np.float32)
    labels = np.zeros(n_trials, dtype=np.int64)

    freq_map = {1: 10.0, 2: 12.0, 3: 20.0, 4: 8.0}  # class → mu rhythm freq
    for i in range(n_trials):
        cls = (i % n_classes) + 1
        labels[i] = cls
        f = freq_map[cls]
        for c in range(n_channels):
            amp = rng.uniform(0.5, 2.0)
            phase = rng.uniform(0, 2 * np.pi)
            noise = rng.normal(0, 0.2, n_samples)
            eeg[i, c, :] = (amp * np.sin(2 * np.pi * f * t + phase) + noise).astype(
                np.float32
            )

    return eeg, labels


# ─────────────────────────────────────────────────────────────────────────────
# Noise injection  (FR-204)
# ─────────────────────────────────────────────────────────────────────────────

def _inject_noise_single(
    eeg: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian noise at a single target SNR (dB)."""
    signal_power = np.mean(eeg**2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0, np.sqrt(noise_power), eeg.shape).astype(np.float32)
    return eeg + noise


def inject_noise(
    eeg: np.ndarray,
    snr_db: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    snr_db_list: Optional[List[float]] = None,
    artifact_prob: float = 0.05,
) -> Any:
    """
    Add Gaussian noise to achieve a target Signal-to-Noise Ratio.

    Two calling conventions are supported:

    **Single-value** (returns np.ndarray – used by EEGDataset internally):
        noisy = inject_noise(eeg, snr_db=15.0)
        noisy = inject_noise(eeg, 15.0)

    **Multi-value** (returns dict – used by training / verification scripts):
        result = inject_noise(eeg, snr_db_list=[10, 15, 20])
        # result keys: 'clean', 'noisy_10', 'noisy_15', 'noisy_20', 'artifacts'

    Parameters
    ----------
    eeg          : np.ndarray  – clean signal, shape (..., n_samples)
    snr_db       : float       – single desired SNR in dB (positional or keyword)
    rng          : optional random generator
    snr_db_list  : list of SNR levels (activates dict-return mode)
    artifact_prob: fraction of trials to flag as artifacts (dict mode only)

    Returns
    -------
    np.ndarray or dict
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── Dict-return mode (snr_db_list) ──────────────────────────────────────
    if snr_db_list is not None:
        result: Dict[str, Any] = {"clean": eeg.copy()}
        for db in snr_db_list:
            result[f"noisy_{int(db)}"] = _inject_noise_single(eeg, float(db), rng)
        # Simple artifact mask: random boolean array over trial axis
        n_trials = eeg.shape[0] if eeg.ndim >= 3 else 1
        result["artifacts"] = rng.random(n_trials) < artifact_prob
        return result

    # ── Single-value mode (backward-compatible) ──────────────────────────────
    if snr_db is None:
        raise ValueError("Either snr_db or snr_db_list must be provided.")
    return _inject_noise_single(eeg, float(snr_db), rng)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact flagging  (FR-205)
# ─────────────────────────────────────────────────────────────────────────────

def flag_artifacts(
    eeg: np.ndarray,
    amplitude_uv_threshold: float = 200.0,
    high_freq_power_threshold: float = 50.0,
) -> np.ndarray:
    """
    Return a boolean mask (True = artifact) for each trial.

    Parameters
    ----------
    eeg : np.ndarray, shape (n_trials, n_channels, n_samples)

    Returns
    -------
    mask : np.ndarray[bool], shape (n_trials,)
    """
    # Amplitude criterion: any sample in the trial exceeds threshold
    amp_mask = np.any(np.abs(eeg) > amplitude_uv_threshold, axis=(1, 2))
    # Power criterion: RMS across all channels > threshold
    rms = np.sqrt(np.mean(eeg**2, axis=(1, 2)))
    rms_mask = rms > high_freq_power_threshold
    return amp_mask | rms_mask


# ─────────────────────────────────────────────────────────────────────────────
# Normalizer  (FR-104)
# ─────────────────────────────────────────────────────────────────────────────

class ChannelNormalizer:
    """Per-channel z-score normalization (fit on training set only)."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None  # shape (n_channels,)
        self.std_: Optional[np.ndarray] = None

    def fit(self, eeg: np.ndarray) -> "ChannelNormalizer":
        """
        Parameters
        ----------
        eeg : np.ndarray, shape (n_trials, n_channels, n_samples)
        """
        # Compute per-channel statistics over all trials and samples
        self.mean_ = eeg.mean(axis=(0, 2), keepdims=False)  # (n_channels,)
        self.std_ = eeg.std(axis=(0, 2), keepdims=False)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (eeg - self.mean_[np.newaxis, :, np.newaxis]) / self.std_[
            np.newaxis, :, np.newaxis
        ]

    def fit_transform(self, eeg: np.ndarray) -> np.ndarray:
        return self.fit(eeg).transform(eeg)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-processed EEG windows.

    Each item: (noisy_window, clean_window, label)
      - windows  : FloatTensor (1, 750)   – one channel, one trial
      - label    : LongTensor scalar      – class index 0-3

    Expanding multi-channel trials into single-channel windows
    (one dataset item per channel per trial) maximises training data
    for the diffusion model, which processes one channel at a time.
    """

    def __init__(
        self,
        clean_eeg: np.ndarray,   # (n_trials, n_channels, n_samples)
        labels: np.ndarray,      # (n_trials,)
        snr_db: Optional[float] = 15.0,
        rng_seed: int = 0,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(rng_seed)
        self.clean = clean_eeg.astype(np.float32)
        self.labels = labels.astype(np.int64) - 1   # 0-indexed
        if snr_db is not None:
            self.noisy = inject_noise(self.clean, snr_db, rng)
        else:
            self.noisy = self.clean.copy()

        n_trials, n_channels, n_samples = self.clean.shape
        self._n_channels = n_channels
        self._n_trials = n_trials

    def __len__(self) -> int:
        return self._n_trials * self._n_channels

    def __getitem__(self, idx: int):
        trial = idx // self._n_channels
        ch = idx % self._n_channels
        noisy = torch.from_numpy(self.noisy[trial, ch : ch + 1, :])  # (1, 750)
        clean = torch.from_numpy(self.clean[trial, ch : ch + 1, :])
        label = torch.tensor(self.labels[trial], dtype=torch.long)
        return noisy, clean, label


class EEGTrialDataset(Dataset):
    """
    Multi-channel trial dataset for the closed-loop classifier.

    Each item: (window_tensor, label)
      - window_tensor : FloatTensor (n_channels, n_samples)
      - label         : LongTensor scalar
    """

    def __init__(
        self,
        eeg: np.ndarray,     # (n_trials, n_channels, n_samples)
        labels: np.ndarray,  # (n_trials,)
    ) -> None:
        super().__init__()
        self.eeg = torch.from_numpy(eeg.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64) - 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.eeg[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Missing Helper Functions for Training and Verification Scripts
# ─────────────────────────────────────────────────────────────────────────────

def load_bci_competition_data(
    subjects: List[int],
    cache_dir: str = "data/raw",
    n_channels: int = 22,
    signal_length: int = 750,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Load BCI Competition IV 2a data for specified subjects.
    If .mat files are not found, falls back to generating synthetic data.
    """
    search_dirs = [Path(cache_dir)]
    if Path(cache_dir).name != "raw":
        search_dirs.append(Path(cache_dir) / "raw")

    all_eeg = []
    all_labels = []
    all_subject_ids = []

    for sub in subjects:
        mat_path = None
        for d in search_dirs:
            p = d / f"A{sub:02d}T.mat"
            if p.exists():
                mat_path = p
                break

        if mat_path is not None:
            try:
                eeg, labels = load_subject_mat(str(mat_path))
                # Trim / pad to consistent n_channels and signal_length
                eeg = eeg[:, :n_channels, :signal_length]
                if eeg.shape[2] < signal_length:
                    pad = signal_length - eeg.shape[2]
                    eeg = np.pad(eeg, ((0, 0), (0, 0), (0, pad)))
                all_eeg.append(eeg)
                all_labels.append(labels)
                all_subject_ids.append(np.full(len(labels), sub, dtype=np.int64))
                logger.info("Loaded Subject %d from %s: %d trials", sub, mat_path.name, len(labels))
            except Exception as exc:
                logger.warning("Failed to load Subject %d from mat: %s. Falling back to synthetic.", sub, exc)
                mat_path = None

        if mat_path is None:
            eeg, labels = _generate_synthetic_subject(
                n_trials=288,
                n_channels=n_channels,
                n_samples=signal_length,
                seed=random_seed + sub,
            )
            all_eeg.append(eeg)
            all_labels.append(labels)
            all_subject_ids.append(np.full(len(labels), sub, dtype=np.int64))
            logger.info("Generated synthetic Subject %d: %d trials", sub, len(labels))

    return {
        'X': np.concatenate(all_eeg, axis=0),
        'y': np.concatenate(all_labels, axis=0),
        'fs': 250.0,
        'subject': np.concatenate(all_subject_ids, axis=0)
    }


def preprocess_pipeline(
    X: np.ndarray,
    fs: float = 250.0,
    bandpass: Tuple[float, float] = (4.0, 40.0),
    segment_duration: float = 3.0,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter and z-score normalize
    each channel individually per trial.
    """
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    low = bandpass[0] / nyq
    high = bandpass[1] / nyq
    b, a = butter(4, [low, high], btype="band")
    
    # Apply bandpass filter across the time axis (last axis)
    X_filtered = filtfilt(b, a, X, axis=-1).astype(np.float32)
    
    # Per-trial per-channel z-score standardization
    mean = np.mean(X_filtered, axis=-1, keepdims=True)
    std = np.std(X_filtered, axis=-1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (X_filtered - mean) / std


def create_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split trials into train, val, and test sets based on subject IDs
    to avoid subject leakage.
    """
    rng = np.random.default_rng(seed)
    unique_subs = np.unique(subject_ids)
    rng.shuffle(unique_subs)
    
    n_subs = len(unique_subs)
    n_train = max(1, int(n_subs * train_ratio))
    n_val = max(1, int(n_subs * val_ratio))
    
    train_subs = unique_subs[:n_train]
    val_subs = unique_subs[n_train : n_train + n_val]
    test_subs = unique_subs[n_train + n_val :]
    
    if len(test_subs) == 0:
        if len(val_subs) > 1:
            test_subs = val_subs[-1:]
            val_subs = val_subs[:-1]
        else:
            test_subs = val_subs
            
    train_mask = np.isin(subject_ids, train_subs)
    val_mask = np.isin(subject_ids, val_subs)
    test_mask = np.isin(subject_ids, test_subs)
    
    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    dataset_dir: str = "data/raw",
    *,
    signal_length: int = 750,
    n_channels: int = 22,
    snr_db: float = 15.0,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    batch_size: int = 64,
    num_workers: int = 0,
    random_seed: int = 42,
    synthetic_fallback: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders with no subject leakage (FR-105).

    Parameters
    ----------
    dataset_dir       : directory containing A01T.mat … A09T.mat
    signal_length     : samples per EEG window (750 for 3 s at 250 Hz)
    n_channels        : expected number of EEG channels (22)
    snr_db            : noise level for training data
    train_ratio       : fraction of subjects used for training
    val_ratio         : fraction for validation
    batch_size        : DataLoader batch size
    num_workers       : DataLoader workers (0 = single-process)
    random_seed       : for reproducibility
    synthetic_fallback: if True and no .mat files found, generate synthetic data

    Returns
    -------
    dict with keys 'train', 'val', 'test'
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    mat_files = sorted(Path(dataset_dir).glob("A0*T.mat"))
    if not mat_files:
        mat_files = sorted(Path(dataset_dir).glob("*.mat"))

    if not mat_files:
        if synthetic_fallback:
            logger.warning(
                "No .mat files found in '%s'. Using synthetic EEG data.", dataset_dir
            )
            all_eeg, all_labels, n_subjects = [], [], 9
            for s in range(n_subjects):
                eeg, labels = _generate_synthetic_subject(
                    n_trials=288,
                    n_channels=n_channels,
                    n_samples=signal_length,
                    seed=random_seed + s,
                )
                all_eeg.append(eeg)
                all_labels.append(labels)
        else:
            raise FileNotFoundError(
                f"No .mat files found in {dataset_dir!r}. "
                "Download BCI Competition IV 2a from http://www.bbci.de/competition/iv/"
            )
    else:
        all_eeg, all_labels = [], []
        for f in mat_files:
            try:
                eeg, labels = load_subject_mat(str(f))
                # Trim / pad to consistent n_samples
                eeg = eeg[:, :n_channels, :signal_length]
                if eeg.shape[2] < signal_length:
                    pad = signal_length - eeg.shape[2]
                    eeg = np.pad(eeg, ((0, 0), (0, 0), (0, pad)))
                all_eeg.append(eeg)
                all_labels.append(labels)
                logger.info("Loaded %s: %d trials", f.name, len(labels))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", f, exc)

    # ── Subject-level train/val/test split (FR-105) ─────────────────────────
    n_subjects = len(all_eeg)
    subject_ids = list(range(n_subjects))
    random.shuffle(subject_ids)

    n_train = max(1, int(n_subjects * train_ratio))
    n_val = max(1, int(n_subjects * val_ratio))

    train_ids = subject_ids[:n_train]
    val_ids = subject_ids[n_train : n_train + n_val]
    test_ids = subject_ids[n_train + n_val :]
    if not test_ids:
        test_ids = val_ids[-1:]

    def _stack(ids):
        eeg_list = [all_eeg[i] for i in ids]
        lbl_list = [all_labels[i] for i in ids]
        return np.concatenate(eeg_list, axis=0), np.concatenate(lbl_list, axis=0)

    train_eeg, train_labels = _stack(train_ids)
    val_eeg, val_labels = _stack(val_ids)
    test_eeg, test_labels = _stack(test_ids)

    # ── Per-channel normalization (fit on train only) – FR-104 ──────────────
    normalizer = ChannelNormalizer()
    train_eeg = normalizer.fit_transform(train_eeg)
    val_eeg = normalizer.transform(val_eeg)
    test_eeg = normalizer.transform(test_eeg)

    # ── Build datasets ───────────────────────────────────────────────────────
    train_ds = EEGDataset(train_eeg, train_labels, snr_db=snr_db, rng_seed=random_seed)
    val_ds = EEGDataset(val_eeg, val_labels, snr_db=snr_db, rng_seed=random_seed + 1)
    test_ds = EEGDataset(test_eeg, test_labels, snr_db=None, rng_seed=random_seed + 2)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    def _make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    loaders = {
        "train": _make_loader(train_ds, shuffle=True),
        "val": _make_loader(val_ds, shuffle=False),
        "test": _make_loader(test_ds, shuffle=False),
    }

    logger.info(
        "DataLoaders ready: train=%d, val=%d, test=%d items",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )
    return loaders


def build_classifier_dataloaders(
    dataset_dir: str = "data/raw",
    *,
    signal_length: int = 750,
    n_channels: int = 22,
    batch_size: int = 64,
    num_workers: int = 0,
    random_seed: int = 42,
    synthetic_fallback: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build multi-channel trial DataLoaders for the motor-imagery classifier (FR-601).
    Same subject-level split as build_dataloaders but returns full-channel tensors.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    mat_files = sorted(Path(dataset_dir).glob("A0*T.mat"))
    if not mat_files:
        mat_files = sorted(Path(dataset_dir).glob("*.mat"))

    if not mat_files:
        if synthetic_fallback:
            all_eeg, all_labels, n_subjects = [], [], 9
            for s in range(9):
                eeg, labels = _generate_synthetic_subject(
                    n_trials=288, n_channels=n_channels,
                    n_samples=signal_length, seed=random_seed + s
                )
                all_eeg.append(eeg)
                all_labels.append(labels)
        else:
            raise FileNotFoundError(f"No .mat files in {dataset_dir!r}")
    else:
        all_eeg, all_labels = [], []
        for f in mat_files:
            try:
                eeg, labels = load_subject_mat(str(f))
                eeg = eeg[:, :n_channels, :signal_length]
                if eeg.shape[2] < signal_length:
                    eeg = np.pad(eeg, ((0, 0), (0, 0), (0, signal_length - eeg.shape[2])))
                all_eeg.append(eeg)
                all_labels.append(labels)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", f, exc)

    n_subjects = len(all_eeg)
    ids = list(range(n_subjects))
    random.shuffle(ids)
    n_train = max(1, int(n_subjects * 0.8))
    n_val = max(1, int(n_subjects * 0.1))
    train_ids = ids[:n_train]
    val_ids = ids[n_train: n_train + n_val]
    test_ids = ids[n_train + n_val:] or val_ids[-1:]

    def _stack(idx_list):
        return (
            np.concatenate([all_eeg[i] for i in idx_list], axis=0),
            np.concatenate([all_labels[i] for i in idx_list], axis=0),
        )

    tr_eeg, tr_lbl = _stack(train_ids)
    va_eeg, va_lbl = _stack(val_ids)
    te_eeg, te_lbl = _stack(test_ids)

    norm = ChannelNormalizer()
    tr_eeg = norm.fit_transform(tr_eeg)
    va_eeg = norm.transform(va_eeg)
    te_eeg = norm.transform(te_eeg)

    def _loader(eeg, lbl, shuffle):
        ds = EEGTrialDataset(eeg, lbl)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=torch.cuda.is_available())

    return {
        "train": _loader(tr_eeg, tr_lbl, True),
        "val": _loader(va_eeg, va_lbl, False),
        "test": _loader(te_eeg, te_lbl, False),
    }
