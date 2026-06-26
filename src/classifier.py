"""
classifier.py – Motor imagery CNN classifier for the closed-loop BCI simulation.

FR-601  4-class motor imagery classifier on preprocessed EEG.
FR-602  Full closed-loop pipeline: preprocess → denoise → extract → classify.
FR-603  Denoising impact measurement: ∆Accuracy per method.
FR-604  Real-time simulation with class/confidence/latency visualization.
FR-605  Artifact handling: skip flagged trials, log rejection rate.

SRS UPGRADE:
  - Accuracy target ≥80% (was ≥75%)
  - End-to-end latency target <50 ms GPU (primary), <500 ms CPU (secondary)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_bandpower_features(
    eeg: np.ndarray,
    fs: float = 250.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Extract log-bandpower features for each EEG channel.

    Parameters
    ----------
    eeg   : np.ndarray, shape (n_trials, n_channels, n_samples)
    fs    : sampling rate
    bands : dict name → (low_hz, high_hz); defaults to delta/theta/alpha/beta/gamma

    Returns
    -------
    np.ndarray, shape (n_trials, n_channels * n_bands)
    """
    if bands is None:
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }

    n_trials, n_ch, n_samp = eeg.shape
    n_bands = len(bands)
    feats = np.zeros((n_trials, n_ch * n_bands), dtype=np.float32)

    freqs = np.fft.rfftfreq(n_samp, d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(eeg, axis=-1)) ** 2) / n_samp

    for tri in range(n_trials):
        for ci in range(n_ch):
            for bi, (_, (lo, hi)) in enumerate(bands.items()):
                mask = (freqs >= lo) & (freqs < hi)
                bp = np.sum(psd[tri, ci, mask]) if mask.any() else 1e-10
                feats[tri, ci * n_bands + bi] = np.log(bp + 1e-10)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# FR-601  Motor imagery CNN classifier
# ─────────────────────────────────────────────────────────────────────────────

class MotorImageryClassifier(nn.Module):
    """
    Compact 1-D CNN classifier for 4-class motor imagery EEG.

    Input: (B, n_channels, n_samples) – full multi-channel EEG window.
    Output: (B, 4) – class logits.

    Designed to achieve ≥80% accuracy on BCI Competition IV 2a
    when the input is properly denoised and band-pass filtered.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 750,
        n_classes: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        # Temporal convolution (across time within each channel)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 11), padding=(0, 5)),
            nn.BatchNorm2d(25),
            nn.ELU(),
        )

        # Spatial convolution (across channels)
        self.spatial = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(n_channels, 1), groups=1),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout / 2),
        )

        # Depthwise separable convolution for further feature extraction
        self.separable = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 11), padding=(0, 5), groups=50),
            nn.Conv2d(100, 100, kernel_size=(1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # Compute flattened feature size
        dummy = torch.zeros(1, 1, n_channels, n_samples)
        dummy = self.temporal(dummy)
        dummy = self.spatial(dummy)
        dummy = self.separable(dummy)
        flat_dim = dummy.view(1, -1).shape[1]

        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, n_channels, n_samples)

        Returns
        -------
        Tensor, shape (B, n_classes) – class logits
        """
        h = x.unsqueeze(1)          # (B, 1, C, T)
        h = self.temporal(h)
        h = self.spatial(h)
        h = self.separable(h)
        h = h.flatten(1)
        return self.classifier(h)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop for classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(
    model: MotorImageryClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "models/classifier",
    early_stop_patience: int = 20,
    device: str = "cuda",
) -> dict:
    """
    Train the motor imagery classifier (FR-601).

    Returns
    -------
    dict with train_losses, val_losses, val_accuracies, best_epoch
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0
    patience_cnt = 0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, correct, total = 0.0, 0, 0
        for eeg, labels in train_loader:
            eeg, labels = eeg.to(dev), labels.to(dev)
            optimizer.zero_grad()
            logits = model(eeg)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        train_losses.append(ep_loss / max(len(train_loader), 1))

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for eeg, labels in val_loader:
                eeg, labels = eeg.to(dev), labels.to(dev)
                logits = model(eeg)
                vl += criterion(logits, labels).item()
                vc += (logits.argmax(1) == labels).sum().item()
                vt += labels.size(0)
        val_acc = vc / max(vt, 1)
        val_losses.append(vl / max(len(val_loader), 1))
        val_accs.append(val_acc)

        logger.info(
            "Classifier Epoch %4d/%d | train_loss=%.4f | val_acc=%.3f",
            epoch, epochs, train_losses[-1], val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_cnt = 0
            torch.save({"model_state": model.state_dict(), "val_acc": val_acc},
                       os.path.join(checkpoint_dir, "best_classifier.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= early_stop_patience:
                logger.info("Classifier early stop at epoch %d.", epoch)
                break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
    }


def evaluate_classifier(
    model: MotorImageryClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
) -> Tuple[float, np.ndarray]:
    """
    Evaluate classifier accuracy and return confusion matrix.

    Returns
    -------
    accuracy : float
    confusion_matrix : np.ndarray, shape (n_classes, n_classes)
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    n_classes = model.classifier.out_features
    conf_mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    correct, total = 0, 0

    with torch.no_grad():
        for eeg, labels in data_loader:
            eeg, labels = eeg.to(dev), labels.to(dev)
            preds = model(eeg).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, g in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                conf_mat[g, p] += 1

    return correct / max(total, 1), conf_mat


# ─────────────────────────────────────────────────────────────────────────────
# FR-602 / FR-603  Closed-loop pipeline + denoising impact
# ─────────────────────────────────────────────────────────────────────────────

class ClosedLoopSimulator:
    """
    Simulates a real-time closed-loop BCI pipeline.

    Pipeline per trial:
      raw EEG → [artifact check] → bandpass filter → denoise → classify
      → record latency + prediction

    FR-602  End-to-end latency < 50 ms (GPU) / <500 ms (CPU).
    FR-603  Accuracy comparison with/without denoising.
    FR-604  Visualization of predictions (plotted after simulation).
    FR-605  Artifact handling: skip flagged trials.
    """

    def __init__(
        self,
        classifier: MotorImageryClassifier,
        denoiser: Optional[Callable] = None,
        device: str = "cuda",
        fs: float = 250.0,
        latency_target_ms: float = 50.0,
    ) -> None:
        self.classifier = classifier
        self.denoiser = denoiser
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fs = fs
        self.latency_target_ms = latency_target_ms
        self.results: List[dict] = []

    def _flag_artifact(self, eeg_window: np.ndarray) -> bool:
        """Return True if the window is an artifact (FR-605)."""
        return bool(np.any(np.abs(eeg_window) > 200.0))

    def _bandpass(self, eeg: np.ndarray) -> np.ndarray:
        """4–40 Hz Butterworth bandpass."""
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [4 / (0.5 * self.fs), 40 / (0.5 * self.fs)], btype="band")
        return filtfilt(b, a, eeg, axis=-1).astype(np.float32)

    def run(
        self,
        eeg_trials: np.ndarray,     # (n_trials, n_channels, n_samples)
        true_labels: np.ndarray,    # (n_trials,) 1-indexed
    ) -> dict:
        """
        Run the closed-loop simulation on all trials.

        Parameters
        ----------
        eeg_trials  : raw EEG windows (n_trials, n_channels, n_samples)
        true_labels : class labels for accuracy computation

        Returns
        -------
        dict with accuracy, mean_latency_ms, artifact_rejection_rate, predictions, latencies
        """
        from baselines import flag_artifacts

        n_trials = eeg_trials.shape[0]
        predictions, latencies = [], []
        n_artifacts = 0

        self.classifier = self.classifier.to(self.device).eval()

        for i in range(n_trials):
            window = eeg_trials[i]  # (n_channels, n_samples)

            # FR-605 Artifact check
            artifact_mask = flag_artifacts(window[np.newaxis], amplitude_uv_threshold=200.0)
            if artifact_mask[0]:
                n_artifacts += 1
                logger.debug("Trial %d flagged as artifact — skipping.", i)
                predictions.append(-1)  # sentinel
                latencies.append(float("nan"))
                continue

            t_start = time.perf_counter()

            # Bandpass
            window_bp = self._bandpass(window)

            # Denoise (optional)
            if self.denoiser is not None:
                x_t = torch.from_numpy(window_bp).unsqueeze(0).to(self.device)  # (1, C, L)
                # Apply denoiser channel-by-channel if it expects (B, 1, L)
                channels_denoised = []
                for c in range(x_t.shape[1]):
                    ch = x_t[:, c : c + 1, :]  # (1, 1, L)
                    with torch.no_grad():
                        ch_d = self.denoiser(ch)
                    channels_denoised.append(ch_d)
                window_denoised = torch.cat(channels_denoised, dim=1)  # (1, C, L)
            else:
                window_denoised = torch.from_numpy(window_bp).unsqueeze(0).to(self.device)

            # Classify
            with torch.no_grad():
                logits = self.classifier(window_denoised)
                pred = int(logits.argmax(1).item())

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            predictions.append(pred)
            latencies.append(latency_ms)

        valid_mask = np.array([p != -1 for p in predictions])
        valid_preds = np.array([p for p in predictions if p != -1])
        valid_labels = (true_labels[valid_mask] - 1)  # 0-indexed
        valid_latencies = np.array([l for l in latencies if not np.isnan(l)])

        accuracy = float(np.mean(valid_preds == valid_labels)) if len(valid_preds) > 0 else 0.0
        mean_lat = float(np.mean(valid_latencies)) if len(valid_latencies) > 0 else float("nan")
        art_rate = n_artifacts / n_trials

        logger.info(
            "Closed-loop: acc=%.3f, mean_lat=%.1f ms, artifact_rate=%.2f",
            accuracy, mean_lat, art_rate,
        )

        return {
            "accuracy": accuracy,
            "mean_latency_ms": mean_lat,
            "p95_latency_ms": float(np.percentile(valid_latencies, 95)) if len(valid_latencies) > 0 else float("nan"),
            "artifact_rejection_rate": art_rate,
            "predictions": predictions,
            "latencies": latencies,
            "n_artifacts": n_artifacts,
            "n_valid_trials": int(valid_mask.sum()),
        }


def measure_denoising_impact(
    classifier: MotorImageryClassifier,
    eeg_trials: np.ndarray,
    true_labels: np.ndarray,
    denoisers: Dict[str, Optional[Callable]],
    device: str = "cuda",
) -> Dict[str, dict]:
    """
    Measure classification accuracy with and without each denoiser.

    FR-603  Report ∆Accuracy (%) for each denoising method.

    Parameters
    ----------
    classifier : trained MotorImageryClassifier
    eeg_trials : (n_trials, n_channels, n_samples)
    true_labels: (n_trials,) 1-indexed
    denoisers  : dict name → callable (or None for baseline no-denoise)

    Returns
    -------
    dict name → result_dict (from ClosedLoopSimulator.run)
    """
    results = {}
    baseline_acc = None

    for name, denoiser in denoisers.items():
        sim = ClosedLoopSimulator(classifier, denoiser=denoiser, device=device)
        result = sim.run(eeg_trials, true_labels)
        results[name] = result
        if name == "no_denoise":
            baseline_acc = result["accuracy"]

    # Compute delta accuracy
    if baseline_acc is not None:
        for name, r in results.items():
            r["delta_accuracy"] = r["accuracy"] - baseline_acc

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FR-604 / FR-702  Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    conf_mat: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    output_path: str = "results/plots/confusion_matrix.png",
) -> None:
    """Plot and save a confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = conf_mat.shape[0]
    if class_names is None:
        class_names = ["Left", "Right", "Feet", "Tongue"][:n]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf_mat, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    thresh = conf_mat.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(conf_mat[i, j]), ha="center", va="center",
                    fontsize=11,
                    color="white" if conf_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved to %s", output_path)


def plot_denoising_impact(
    impact_results: Dict[str, dict],
    output_path: str = "results/plots/denoising_impact.png",
) -> None:
    """Bar chart showing classification accuracy per denoising method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    names = list(impact_results.keys())
    accs = [impact_results[n]["accuracy"] for n in names]
    deltas = [impact_results[n].get("delta_accuracy", 0) for n in names]
    colors = ["#9E9E9E" if n == "no_denoise" else "#2196F3" for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(names, accs, color=colors, edgecolor="white")
    ax1.axhline(0.80, color="red", linestyle="--", linewidth=1.2, label="Target (80%)")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Classification Accuracy", fontsize=11)
    ax1.set_title("Accuracy by Denoising Method", fontsize=12, fontweight="bold")
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(names[1:], deltas[1:], color="#4CAF50", edgecolor="white")
    ax2.axhline(0, color="gray", linewidth=1)
    ax2.set_ylabel("∆ Accuracy vs No-Denoise", fontsize=11)
    ax2.set_title("Denoising Impact (∆ Accuracy)", fontsize=12, fontweight="bold")
    ax2.set_xticklabels(names[1:], rotation=30, ha="right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Denoising impact plot saved to %s", output_path)
