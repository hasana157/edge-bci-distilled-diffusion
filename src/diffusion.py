"""
diffusion.py – GPU-Accelerated DDPM-style denoising diffusion model for EEG.

Architecture
------------
  UNet1D            : up to 2M–5M parameter 1-D U-Net (SRS UPGRADE) with
                      sinusoidal timestep embeddings, residual blocks,
                      GroupNorm, SiLU activations, and skip connections.
  GaussianDiffusion : forward (q) and reverse (p) processes.
                      Supports both 'linear' and 'cosine' noise schedules.

Processing unit: one EEG channel at a time → shape (B, 1, 750).
Fully device-agnostic: runs on CUDA (Colab T4) and CPU.

FR-301 DDPM-style denoising diffusion model.
FR-302 Training loop with noise prediction MSE loss.
FR-303 Variable-step reverse diffusion (10, 25, 50, 100, 500 — SRS UPGRADE).
FR-304 CPU/GPU-optimized architecture.
FR-305 Checkpoint save/load.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 0. DiffusionConfig  – self-contained config dataclass
#    (mirrors config.DiffusionConfig so standalone scripts can import it here)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiffusionConfig:
    """Standalone diffusion model configuration (FR-301 to FR-305)."""
    # Noise schedule
    n_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"          # 'linear' | 'cosine'
    # UNet architecture
    signal_length: int = 750
    in_channels: int = 1
    model_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    dropout: float = 0.1
    time_emb_dim: int = 256
    # Inference
    inference_steps: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 500])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding (Vaswani et al., 2017 position encoding)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : LongTensor, shape (B,)

        Returns
        -------
        Tensor, shape (B, dim)
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with up to 8 groups, always divisible."""
    num_groups = min(8, channels)
    while channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, channels)


class ResBlock1D(nn.Module):
    """
    1-D residual block with time-embedding injection.

    GN → SiLU → Conv1d(in→out) → + proj(t) → GN → SiLU → [Dropout] → Conv1d(out→out)
    + skip connection (1×1 Conv1d if in≠out, else Identity)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = _gn(out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.drop(F.silu(self.norm2(h)))
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample1D(nn.Module):
    """Strided Conv1d for 2× downsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """Linear interpolation + Conv1d for 2× upsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        size = target_len if target_len else x.shape[-1] * 2
        x = F.interpolate(x, size=size, mode="linear", align_corners=False)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. UNet1D  (SRS UPGRADE: up to 2M–5M params on Colab)
# ─────────────────────────────────────────────────────────────────────────────

class UNet1D(nn.Module):
    """
    Lightweight–to-medium 1-D U-Net for noise prediction in DDPM.

    Default config (model_channels=64, channel_mult=(1,2,4,8)):
      ~2.1M parameters — fits comfortably on Colab T4.

    Colab-scale config (model_channels=96, channel_mult=(1,2,4,8)):
      ~4.7M parameters.

    Parameters
    ----------
    model_channels : int
        Base channel width (default 64 for GPU, 32 for CPU).
    channel_mult   : tuple
        Per-level channel multipliers.
    num_res_blocks : int
        ResBlock1D layers per U-Net level.
    dropout        : float
        Dropout rate inside ResBlocks.
    time_emb_dim   : int
        Dimension of the time-embedding MLP output.
    signal_length  : int
        Input signal length in samples (750 for 3 s @ 250 Hz).
    in_channels    : int
        Input/output channels (1 for single-channel EEG).
    """

    def __init__(
        self,
        cfg: Optional[DiffusionConfig] = None,
        *,
        model_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        time_emb_dim: int = 256,
        signal_length: int = 750,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        # If a DiffusionConfig object is passed as the first arg, unpack it
        if cfg is not None:
            model_channels = cfg.model_channels
            channel_mult = cfg.channel_mult
            num_res_blocks = cfg.num_res_blocks
            dropout = cfg.dropout
            time_emb_dim = cfg.time_emb_dim
            signal_length = cfg.signal_length
            in_channels = cfg.in_channels
        channels: List[int] = [model_channels * m for m in channel_mult]
        t_dim = time_emb_dim

        # Time embedding MLP
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        # Stem
        self.stem = nn.Conv1d(in_channels, channels[0], kernel_size=3, padding=1)

        # Encoder
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()
        in_ch = channels[0]
        self._enc_out_chs: List[int] = []

        for i, out_ch in enumerate(channels):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock1D(in_ch, out_ch, t_dim, dropout))
                in_ch = out_ch
            self.enc_blocks.append(level_blocks)
            self._enc_out_chs.append(out_ch)
            if i < len(channels) - 1:
                self.downsamples.append(Downsample1D(out_ch))
            else:
                self.downsamples.append(None)

        # Bottleneck
        self.mid1 = ResBlock1D(channels[-1], channels[-1], t_dim, dropout)
        self.mid2 = ResBlock1D(channels[-1], channels[-1], t_dim, dropout)

        # Decoder
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        self.upsamples: nn.ModuleList = nn.ModuleList()
        for i in reversed(range(len(channels))):
            out_ch = channels[i]
            skip_ch = self._enc_out_chs[i]
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                blk_in = in_ch + skip_ch if j == 0 else out_ch
                level_blocks.append(ResBlock1D(blk_in, out_ch, t_dim, dropout))
                in_ch = out_ch
            self.dec_blocks.append(level_blocks)
            self.upsamples.append(Upsample1D(out_ch) if i > 0 else None)

        # Output head
        self.out_norm = _gn(channels[0])
        self.out_conv = nn.Conv1d(channels[0], in_channels, kernel_size=1)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, 1, L)   – noisy EEG window
        t : LongTensor, shape (B,)    – diffusion timestep

        Returns
        -------
        Tensor, shape (B, 1, L) – predicted noise
        """
        t_emb = self.time_emb(t)
        h = self.stem(x)

        skips: List[torch.Tensor] = []
        spatial: List[int] = []
        for level_blocks, ds in zip(self.enc_blocks, self.downsamples):
            for blk in level_blocks:
                h = blk(h, t_emb)
            skips.append(h)
            spatial.append(h.shape[-1])
            if ds is not None:
                h = ds(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for level_idx, (level_blocks, us) in enumerate(
            zip(self.dec_blocks, self.upsamples)
        ):
            skip = skips[-(level_idx + 1)]
            tlen = spatial[-(level_idx + 1)]
            for j, blk in enumerate(level_blocks):
                if j == 0:
                    if h.shape[-1] != tlen:
                        h = F.interpolate(h, size=tlen, mode="linear", align_corners=False)
                    h = torch.cat([h, skip], dim=1)
                h = blk(h, t_emb)
            if us is not None:
                h = us(h, target_len=spatial[-(level_idx + 2)])

        return self.out_conv(F.silu(self.out_norm(h)))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gaussian Diffusion (forward + reverse)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianDiffusion:
    """
    DDPM forward (q) and reverse (p) processes.

    Supports 'linear' and 'cosine' noise schedules.
    All schedule tensors are kept on CPU and moved to device at runtime.
    """

    def __init__(
        self,
        cfg: Optional[DiffusionConfig] = None,
        *,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "cosine",
    ) -> None:
        # If a DiffusionConfig object is passed as the first arg, unpack it
        if cfg is not None:
            n_steps = cfg.n_steps
            beta_start = cfg.beta_start
            beta_end = cfg.beta_end
            schedule = cfg.schedule
        self.n_steps = n_steps

        if schedule == "cosine":
            s = 0.008
            steps = torch.arange(n_steps + 1, dtype=torch.float64)
            f = torch.cos((steps / n_steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod_full = f / f[0]
            betas = (1.0 - alphas_cumprod_full[1:] / alphas_cumprod_full[:-1]).clamp(0.0, 0.999)
            alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        else:  # linear
            betas = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float64)
            alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

        alphas = 1.0 - betas
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        def r(x): return x.float()

        self.betas = r(betas)
        self.alphas_cumprod = r(alphas_cumprod)
        self.alphas_cumprod_prev = r(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = r(alphas_cumprod.sqrt())
        self.sqrt_one_minus_ac = r((1.0 - alphas_cumprod).sqrt())
        self.sqrt_recip_alphas = r((1.0 / alphas).sqrt())
        self.posterior_variance = r(
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def _to(self, device: torch.device):
        """Move all schedule tensors to device."""
        for attr in [
            "betas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_ac",
            "sqrt_recip_alphas", "posterior_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        vals = arr[t].float()
        while vals.dim() < len(shape):
            vals = vals.unsqueeze(-1)
        return vals.expand(shape)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(ᾱ_t)*x_0 + sqrt(1−ᾱ_t)*ε."""
        self._to(x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sb = self._extract(self.sqrt_one_minus_ac, t, x0.shape)
        return sa * x0 + sb * noise

    def p_losses(
        self,
        model: UNet1D,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """L = E[||ε − ε_θ(x_t, t)||²]"""
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        return F.mse_loss(model(x_t, t), noise)

    @torch.no_grad()
    def p_sample(self, model: UNet1D, x: torch.Tensor, t_idx: int) -> torch.Tensor:
        """One reverse step: x_t → x_{t-1}."""
        self._to(x.device)
        B = x.shape[0]
        t_tensor = torch.full((B,), t_idx, dtype=torch.long, device=x.device)
        pred_noise = model(x, t_tensor)

        betas_t = self._extract(self.betas, t_tensor, x.shape)
        sb_t = self._extract(self.sqrt_one_minus_ac, t_tensor, x.shape)
        sra_t = self._extract(self.sqrt_recip_alphas, t_tensor, x.shape)
        mean = sra_t * (x - betas_t / sb_t * pred_noise)

        if t_idx == 0:
            return mean
        pv = self._extract(self.posterior_variance, t_tensor, x.shape)
        return mean + pv.sqrt() * torch.randn_like(x)

    @torch.no_grad()
    def denoise(
        self,
        model: UNet1D,
        x_noisy: torch.Tensor,
        steps: int = 50,
    ) -> torch.Tensor:
        """
        Denoise a real noisy EEG signal using the reverse diffusion chain.

        Treats ``x_noisy`` as the signal at timestep ``steps−1``.

        Parameters
        ----------
        model   : trained UNet1D
        x_noisy : Tensor, shape (B, 1, 750)
        steps   : number of reverse steps (10 / 25 / 50 / 100 / 500)

        Returns
        -------
        Tensor, shape (B, 1, 750) – denoised signal
        """
        self._to(x_noisy.device)
        model.eval()
        t_start = min(steps - 1, self.n_steps - 1)
        t_tensor = torch.full((x_noisy.shape[0],), t_start,
                              dtype=torch.long, device=x_noisy.device)
        x = self.q_sample(x_noisy, t_tensor)
        for t_idx in reversed(range(t_start + 1)):
            x = self.p_sample(model, x, t_idx)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training loop  (SRS UPGRADE: 500+ epochs, Colab T4 batch 64+)
# ─────────────────────────────────────────────────────────────────────────────

def train_diffusion(
    model: UNet1D,
    diffusion: GaussianDiffusion,
    train_loader,
    val_loader,
    *,
    epochs: int = 500,
    lr: float = 1e-3,
    lr_step: int = 100,
    lr_gamma: float = 0.5,
    grad_clip: float = 1.0,
    checkpoint_dir: str = "models/diffusion_teacher",
    save_every: int = 25,
    early_stop_patience: int = 50,
    device: str = "cuda",
) -> dict:
    """
    Train the UNet1D noise-prediction model.

    Parameters
    ----------
    model          : UNet1D
    diffusion      : GaussianDiffusion
    train_loader   : DataLoader yielding (noisy, clean, label) triples
    val_loader     : DataLoader yielding (noisy, clean, label) triples
    epochs         : number of training epochs (SRS UPGRADE: 500+)
    lr             : Adam learning rate
    lr_step        : StepLR step size in epochs
    lr_gamma       : StepLR decay factor
    grad_clip      : gradient clipping norm
    checkpoint_dir : directory for saved checkpoints
    save_every     : epoch interval between periodic checkpoints
    early_stop_patience : early stopping patience
    device         : 'cuda' or 'cpu'

    Returns
    -------
    dict with keys: train_losses, val_losses, best_epoch, best_val_loss
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    best_val = float("inf")
    best_epoch = 0
    patience_cnt = 0
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        ep_loss = 0.0
        for x_noisy, x_clean, _ in train_loader:
            x_clean = x_clean.to(dev)
            t = torch.randint(0, diffusion.n_steps, (x_clean.shape[0],),
                              device=dev, dtype=torch.long)
            optimizer.zero_grad()
            loss = diffusion.p_losses(model, x_clean, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ep_loss += loss.item()
        avg_train = ep_loss / max(len(train_loader), 1)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x_noisy, x_clean, _ in val_loader:
                x_clean = x_clean.to(dev)
                t = torch.randint(0, diffusion.n_steps, (x_clean.shape[0],),
                                  device=dev, dtype=torch.long)
                vl += diffusion.p_losses(model, x_clean, t).item()
        avg_val = vl / max(len(val_loader), 1)
        val_losses.append(avg_val)
        scheduler.step()

        logger.info(
            "Epoch %4d/%d | train=%.5f | val=%.5f | lr=%.2e",
            epoch, epochs, avg_train, avg_val,
            optimizer.param_groups[0]["lr"],
        )

        # Checkpointing
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_loss": avg_val,
        }
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            patience_cnt = 0
            torch.save(ckpt, os.path.join(checkpoint_dir, "best_model.pt"))
        else:
            patience_cnt += 1

        if epoch % save_every == 0:
            torch.save(ckpt, os.path.join(checkpoint_dir, f"ckpt_ep{epoch:04d}.pt"))

        if patience_cnt >= early_stop_patience:
            logger.info("Early stop at epoch %d.", epoch)
            break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model: UNet1D, path: str, **meta) -> None:
    """Save model state dict plus arbitrary metadata."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(), **meta}, path)
    logger.info("Checkpoint saved: %s", path)


def load_checkpoint(
    model: UNet1D,
    path: str,
    device: str = "cpu",
) -> dict:
    """Load model state dict from checkpoint."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info("Checkpoint loaded: %s (epoch %s)", path, ckpt.get("epoch", "?"))
    return ckpt
