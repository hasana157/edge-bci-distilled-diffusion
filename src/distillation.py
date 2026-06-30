"""
distillation.py – Knowledge distillation framework for EEG denoising.

FR-401  Knowledge distillation: student mimics frozen diffusion teacher.
FR-402  CNN student architecture (<100K params, <50ms CPU inference).
FR-403  Autoencoder student (bottleneck 32–64 dim).
FR-404  Temperature-scaled KL divergence loss (T=4, 75% soft + 25% hard).
FR-405  Hyperparameter sweep over LR and alpha.
FR-406  Consistency distillation (single-step) – NEW (SRS ADD).
FR-350  ONNX export for edge simulation – NEW (SRS ADD).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion import GaussianDiffusion, UNet1D

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FR-402  CNN Student
# ─────────────────────────────────────────────────────────────────────────────

class CNNStudent(nn.Module):
    """
    Lightweight 1-D CNN student model.

    Architecture
    ------------
    Conv1d(1→32, k5) + ReLU + MaxPool(2)
    Conv1d(32→64, k5) + ReLU + MaxPool(2)
    Flatten → Linear(64*(L//4), 256) + ReLU + Dropout(0.2)
    Linear(256, L)

    Target: <100K params, <50ms inference on CPU.

    Parameters
    ----------
    signal_length : int
        Input signal length in samples (default 750).
    """

    def __init__(self, signal_length: int = 750) -> None:
        super().__init__()
        self.signal_length = signal_length
        L2 = signal_length // 2
        L4 = signal_length // 4

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * L4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, signal_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, 1, L)

        Returns
        -------
        Tensor, shape (B, 1, L)
        """
        h = self.encoder(x)
        out = self.fc(h)
        return out.unsqueeze(1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# FR-403  Autoencoder Student
# ─────────────────────────────────────────────────────────────────────────────

class AutoencoderStudent(nn.Module):
    """
    Convolutional Autoencoder student with bottleneck compression.

    Architecture
    ------------
    Encoder : Conv1d(1→16, s2) → Conv1d(16→32, s2) → Linear(latent_dim)
    Decoder : Linear(latent_dim→32*L4) → ConvTranspose1d(32→16) → ConvTranspose1d(16→1)

    Target: ~45K params, ultra-lightweight.

    Parameters
    ----------
    signal_length : int  – input length in samples (750)
    latent_dim    : int  – bottleneck dimension (32–64 per FR-403)
    """

    def __init__(self, signal_length: int = 750, latent_dim: int = 64) -> None:
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        L4 = signal_length // 4

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.enc_linear = nn.Linear(32 * L4, latent_dim)

        self.dec_linear = nn.Linear(latent_dim, 32 * L4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )
        self._L4 = L4

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        return self.enc_linear(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_linear(z).view(z.shape[0], 32, self._L4)
        out = self.decoder_conv(h)
        # Trim / pad to exact signal length
        if out.shape[-1] > self.signal_length:
            out = out[:, :, : self.signal_length]
        elif out.shape[-1] < self.signal_length:
            out = F.pad(out, (0, self.signal_length - out.shape[-1]))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, 1, L)

        Returns
        -------
        Tensor, shape (B, 1, L)
        """
        return self.decode(self.encode(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# FR-406  Consistency Student (single-step denoising)  – NEW
# ─────────────────────────────────────────────────────────────────────────────

class ConsistencyStudent(nn.Module):
    """
    Consistency distillation student: maps noisy EEG directly to clean EEG
    in a single forward pass (no iterative reverse diffusion needed).

    Architecture: slightly larger CNN with residual connections for better
    single-step quality.

    FR-406 – Consistency model distillation (SRS ADD).
    """

    def __init__(self, signal_length: int = 750) -> None:
        super().__init__()
        self.signal_length = signal_length

        self.net = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            # Block 3
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            # Block 4
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            # Output
            nn.Conv1d(64, 1, kernel_size=1),
        )
        self.skip = nn.Conv1d(1, 1, kernel_size=1)  # residual skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# FR-404  Distillation Loss
# ─────────────────────────────────────────────────────────────────────────────

def distillation_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    clean: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """
    Combined distillation loss.

    L = alpha * KL(softmax(teacher/T), softmax(student/T)) * T²
      + (1 - alpha) * MSE(student, clean)

    Parameters
    ----------
    student_out : Tensor, shape (B, 1, L)
    teacher_out : Tensor, shape (B, 1, L)
    clean       : Tensor, shape (B, 1, L) – hard target
    temperature : float – T for soft targets (FR-404, T=4)
    alpha       : float – soft target weight (FR-404, 0.75)

    Returns
    -------
    Tensor scalar loss
    """
    T = temperature
    # Flatten to (B, L) for softmax
    s_flat = student_out.view(student_out.shape[0], -1)
    t_flat = teacher_out.view(teacher_out.shape[0], -1)

    s_soft = F.log_softmax(s_flat / T, dim=-1)
    t_soft = F.softmax(t_flat / T, dim=-1)

    kl_loss = F.kl_div(s_soft, t_soft, reduction="batchmean") * (T**2)
    hard_loss = F.mse_loss(student_out, clean)

    return alpha * kl_loss + (1.0 - alpha) * hard_loss


# ─────────────────────────────────────────────────────────────────────────────
# Core distillation training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_student(
    student: nn.Module,
    teacher: UNet1D,
    diffusion: GaussianDiffusion,
    train_loader,
    val_loader,
    *,
    epochs: int = 75,
    max_epochs_today: Optional[int] = None,
    lr: float = 5e-4,
    lr_step: int = 100,
    lr_gamma: float = 0.5,
    temperature: float = 4.0,
    alpha: float = 0.75,
    teacher_steps: int = 25,
    checkpoint_dir: str = "/content/drive/MyDrive/ebc_checkpoints/distilled",
    model_name: str = "student",
    save_every: int = 5,
    early_stop_patience: int = 50,
    device: str = "cuda",
) -> dict:
    """
    Train a student model via knowledge distillation from the frozen teacher.

    Parameters
    ----------
    student         : student nn.Module (CNNStudent / AutoencoderStudent / ConsistencyStudent)
    teacher         : frozen UNet1D teacher
    diffusion       : GaussianDiffusion used by the teacher for inference
    train_loader    : DataLoader yielding (noisy, clean, label) triples
    val_loader      : DataLoader yielding (noisy, clean, label) triples
    epochs          : training epochs
    lr              : Adam learning rate
    temperature     : KL temperature (T=4 per FR-404)
    alpha           : soft target weight (0.75 per FR-404)
    teacher_steps   : reverse diffusion steps used for teacher inference
    checkpoint_dir  : base directory for checkpoints
    model_name      : sub-folder / prefix for this student's checkpoints
    device          : 'cuda' or 'cpu'

    Returns
    -------
    dict with keys: train_losses, val_losses, best_epoch, best_val_loss
    """
    save_dir = os.path.join(checkpoint_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(dev).eval()
    student = student.to(dev)

    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    best_val = float("inf")
    best_epoch = 0
    start_epoch = 1
    patience_cnt = 0
    train_losses: List[float] = []
    val_losses: List[float] = []

    # Auto-resume logic
    latest_ckpt_path = os.path.join(save_dir, "latest_model.pt")
    if os.path.exists(latest_ckpt_path):
        logger.info("[%s] Resuming from checkpoint: %s", model_name, latest_ckpt_path)
        ckpt = torch.load(latest_ckpt_path, map_location=dev)
        student.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        best_epoch = ckpt.get("best_epoch", 0)
        patience_cnt = ckpt.get("patience_cnt", 0)
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        logger.info("[%s] Resumed at epoch %d. Best val so far: %.5f", model_name, start_epoch, best_val)
    
    target_epochs = epochs
    if max_epochs_today is not None:
        target_epochs = min(epochs, start_epoch + max_epochs_today - 1)
        logger.info("[%s] Training %d new epochs today (up to epoch %d).", model_name, max_epochs_today, target_epochs)

    for epoch in range(start_epoch, target_epochs + 1):
        student.train()
        ep_loss = 0.0
        for x_noisy, x_clean, _ in train_loader:
            x_noisy = x_noisy.to(dev)
            x_clean = x_clean.to(dev)

            # Generate teacher soft targets
            with torch.no_grad():
                teacher_out = diffusion.denoise(teacher, x_noisy, steps=teacher_steps)

            student_out = student(x_noisy)
            loss = distillation_loss(student_out, teacher_out, x_clean, temperature, alpha)

            optimizer.zero_grad()
            loss.backward()
            student_out_detach = student_out.detach()
            optimizer.step()
            ep_loss += loss.item()

        avg_train = ep_loss / max(len(train_loader), 1)
        train_losses.append(avg_train)

        # Validation
        student.eval()
        vl = 0.0
        with torch.no_grad():
            for x_noisy, x_clean, _ in val_loader:
                x_noisy = x_noisy.to(dev)
                x_clean = x_clean.to(dev)
                teacher_out = diffusion.denoise(teacher, x_noisy, steps=teacher_steps)
                student_out = student(x_noisy)
                vl += distillation_loss(student_out, teacher_out, x_clean, temperature, alpha).item()
        avg_val = vl / max(len(val_loader), 1)
        val_losses.append(avg_val)
        scheduler.step()

        logger.info(
            "[%s] Epoch %4d/%d | train=%.5f | val=%.5f",
            model_name, epoch, epochs, avg_train, avg_val,
        )

        ckpt = {
            "epoch": epoch,
            "model_state": student.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss": avg_val,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "patience_cnt": patience_cnt,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            patience_cnt = 0
            torch.save(ckpt, os.path.join(save_dir, "best_model.pt"))
        else:
            patience_cnt += 1

        # Always save latest_model.pt every epoch for distillation since it's slow
        torch.save(ckpt, os.path.join(save_dir, "latest_model.pt"))
        logger.info("[%s] Saved latest checkpoint at epoch %d to Drive.", model_name, epoch)

        if epoch % save_every == 0:
            pass # We could keep historical checkpoints here if we wanted

        if patience_cnt >= early_stop_patience:
            logger.info("Early stop [%s] at epoch %d.", model_name, epoch)
            break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FR-405  Hyperparameter sweep
# ─────────────────────────────────────────────────────────────────────────────

def hyperparameter_sweep(
    student_cls,
    teacher: UNet1D,
    diffusion: GaussianDiffusion,
    train_loader,
    val_loader,
    lr_values: List[float],
    alpha_values: List[float],
    *,
    sweep_epochs: int = 50,
    device: str = "cuda",
    checkpoint_dir: str = "models/distilled/sweep",
) -> List[dict]:
    """
    Grid sweep over (lr, alpha) hyperparameters.

    Parameters
    ----------
    student_cls : class – callable that instantiates a fresh student
    lr_values   : list of learning rates to try
    alpha_values: list of alpha (soft-target weight) values to try
    sweep_epochs: short training run per config

    Returns
    -------
    List of result dicts sorted by best_val_loss ascending.
    """
    results = []
    for lr in lr_values:
        for alpha in alpha_values:
            name = f"lr{lr}_a{alpha}"
            logger.info("Sweep: %s", name)
            student = student_cls()
            res = train_student(
                student, teacher, diffusion, train_loader, val_loader,
                epochs=sweep_epochs, lr=lr, alpha=alpha,
                model_name=name,
                checkpoint_dir=checkpoint_dir,
                device=device,
                early_stop_patience=sweep_epochs,
            )
            res["lr"] = lr
            res["alpha"] = alpha
            results.append(res)

    results.sort(key=lambda r: r["best_val_loss"])
    logger.info("Best sweep config: lr=%s, alpha=%s, val_loss=%.5f",
                results[0]["lr"], results[0]["alpha"], results[0]["best_val_loss"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FR-350  ONNX Export  – NEW
# ─────────────────────────────────────────────────────────────────────────────

def export_to_onnx(
    model: nn.Module,
    export_path: str,
    signal_length: int = 750,
    batch_size: int = 1,
    opset_version: int = 17,
) -> None:
    """
    Export a trained student model to ONNX for edge simulation.

    FR-350: Export distilled model to ONNX format, run inference,
    report latency and memory (edge proxy deliverable).

    Parameters
    ----------
    model         : trained student model (any of CNN/AE/Consistency)
    export_path   : output .onnx file path
    signal_length : input signal length
    batch_size    : dummy batch size for tracing
    opset_version : ONNX opset (17 = latest widely supported)
    """
    os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
    model.eval().cpu()
    dummy = torch.randn(batch_size, 1, signal_length)

    torch.onnx.export(
        model,
        dummy,
        export_path,
        opset_version=opset_version,
        input_names=["eeg_noisy"],
        output_names=["eeg_denoised"],
        dynamic_axes={
            "eeg_noisy": {0: "batch_size"},
            "eeg_denoised": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(export_path) / (1024**2)
    logger.info("ONNX model exported to %s (%.2f MB)", export_path, size_mb)


def benchmark_onnx(
    onnx_path: str,
    signal_length: int = 750,
    n_iterations: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Run ONNX Runtime CPU inference benchmark.

    FR-350 – Simulates edge (CPU) inference after ONNX export.

    Returns
    -------
    dict with keys: mean_ms, std_ms, min_ms, max_ms, p95_ms
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required. pip install onnxruntime")

    import time
    import numpy as np

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 1, signal_length).astype(np.float32)
    iname = sess.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        sess.run(None, {iname: dummy})

    # Measurement
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        sess.run(None, {iname: dummy})
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    result = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }
    logger.info("ONNX CPU benchmark: mean=%.2f ms, p95=%.2f ms", result["mean_ms"], result["p95_ms"])
    return result
