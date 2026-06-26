"""
diffusion.py – Lightweight DDPM-style denoising diffusion model for EEG.

Architecture
------------
  UNet1D   : ~450K parameter 1-D U-Net with sinusoidal timestep embeddings,
             residual blocks, GroupNorm, and SiLU activations.
  GaussianDiffusion : forward (q) and reverse (p) diffusion processes.

Processing unit: one EEG channel at a time  →  shape (B, 1, 750).
All code is CPU-safe; no CUDA-specific calls are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiffusionConfig:
    """Hyper-parameters for the diffusion process and UNet architecture."""

    # Noise schedule
    n_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = 'linear'          # 'linear' | 'cosine'

    # Signal dimensions
    signal_length: int = 750          # samples per channel window
    in_channels: int = 1              # channels fed to the UNet

    # UNet architecture
    model_channels: int = 32          # base feature width
    channel_mult: Tuple[int, ...] = (1, 2, 4)   # → 32, 64, 128 channels
    num_res_blocks: int = 2
    dropout: float = 0.0
    time_emb_dim: int = 128           # dimension of the time embedding MLP


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Building blocks
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
        t : LongTensor, shape (B,)  – diffusion timestep indices

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

    Architecture
    ------------
    GN → SiLU → Conv1d(in→out) → + proj(t) → GN → SiLU → Conv1d(out→out)
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
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None]   # broadcast over time axis
        h = self.dropout(F.silu(self.norm2(h)))
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
        x = F.interpolate(x, size=target_len if target_len else x.shape[-1] * 2,
                          mode='linear', align_corners=False)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  UNet1D
# ─────────────────────────────────────────────────────────────────────────────

class UNet1D(nn.Module):
    """
    Lightweight 1-D U-Net for noise prediction in DDPM.

    Target parameter count: ~450K.

    Parameters
    ----------
    cfg : DiffusionConfig
        Architecture hyperparameters.
    """

    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        base = cfg.model_channels
        mults = cfg.channel_mult        # e.g. (1, 2, 4)
        channels: List[int] = [base * m for m in mults]   # [32, 64, 128]
        t_dim = cfg.time_emb_dim

        # ── Time embedding MLP ──────────────────────────────────────────────
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(base),
            nn.Linear(base, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        # ── Stem: project input to first feature width ───────────────────────
        self.stem = nn.Conv1d(cfg.in_channels, channels[0], kernel_size=3, padding=1)

        # ── Encoder ─────────────────────────────────────────────────────────
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()
        in_ch = channels[0]
        self._enc_out_chs: List[int] = []

        for i, out_ch in enumerate(channels):
            level_blocks = nn.ModuleList()
            for j in range(cfg.num_res_blocks):
                level_blocks.append(
                    ResBlock1D(in_ch, out_ch, t_dim, cfg.dropout)
                )
                in_ch = out_ch
            self.enc_blocks.append(level_blocks)
            self._enc_out_chs.append(out_ch)
            if i < len(channels) - 1:
                self.downsamples.append(Downsample1D(out_ch))
            else:
                self.downsamples.append(None)   # no downsample at bottleneck

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.mid_block1 = ResBlock1D(channels[-1], channels[-1], t_dim, cfg.dropout)
        self.mid_block2 = ResBlock1D(channels[-1], channels[-1], t_dim, cfg.dropout)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        self.upsamples: nn.ModuleList = nn.ModuleList()
        for i in reversed(range(len(channels))):
            out_ch = channels[i]
            skip_ch = self._enc_out_chs[i]
            level_blocks = nn.ModuleList()
            for j in range(cfg.num_res_blocks):
                # first block gets skip concatenation → double input channels
                blk_in = in_ch + skip_ch if j == 0 else out_ch
                level_blocks.append(
                    ResBlock1D(blk_in, out_ch, t_dim, cfg.dropout)
                )
                in_ch = out_ch
            self.dec_blocks.append(level_blocks)
            if i > 0:
                self.upsamples.append(Upsample1D(out_ch))
            else:
                self.upsamples.append(None)

        # ── Output head ─────────────────────────────────────────────────────
        self.out_norm = _gn(channels[0])
        self.out_conv = nn.Conv1d(channels[0], cfg.in_channels, kernel_size=1)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, 1, 750) – noisy input
        t : LongTensor, shape (B,)    – diffusion timestep

        Returns
        -------
        Tensor, shape (B, 1, 750) – predicted noise
        """
        t_emb = self.time_emb(t)       # (B, t_dim)

        # Stem
        h = self.stem(x)               # (B, ch0, L)

        # Encoder – keep track of skip features and spatial sizes
        skips: List[torch.Tensor] = []
        spatial_sizes: List[int] = []
        for level_idx, (level_blocks, ds) in enumerate(
            zip(self.enc_blocks, self.downsamples)
        ):
            for blk in level_blocks:
                h = blk(h, t_emb)
            skips.append(h)
            spatial_sizes.append(h.shape[-1])
            if ds is not None:
                h = ds(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for level_idx, (level_blocks, us) in enumerate(
            zip(self.dec_blocks, self.upsamples)
        ):
            skip = skips[-(level_idx + 1)]
            target_len = spatial_sizes[-(level_idx + 1)]
            for j, blk in enumerate(level_blocks):
                if j == 0:
                    # Align spatial size before concatenation
                    if h.shape[-1] != target_len:
                        h = F.interpolate(h, size=target_len, mode='linear',
                                          align_corners=False)
                    h = torch.cat([h, skip], dim=1)
                h = blk(h, t_emb)
            if us is not None:
                up_target = spatial_sizes[-(level_idx + 2)]
                h = us(h, target_len=up_target)

        # Output
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Gaussian Diffusion (forward + reverse processes)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianDiffusion:
    """
    Implements the DDPM forward (q) and reverse (p) processes.

    Schedules
    ---------
    - **linear**  : β linearly spaced from β_start to β_end
    - **cosine**  : cos² schedule (Nichol & Dhariwal, 2021)

    Usage
    -----
    >>> cfg = DiffusionConfig()
    >>> gd  = GaussianDiffusion(cfg)
    >>> model = UNet1D(cfg)
    >>> x0 = torch.randn(4, 1, 750)
    >>> t  = torch.randint(0, cfg.n_steps, (4,))
    >>> noise = torch.randn_like(x0)
    >>> xt = gd.q_sample(x0, t, noise)
    >>> pred = model(xt, t)
    >>> loss = F.mse_loss(pred, noise)
    """

    def __init__(self, cfg: DiffusionConfig) -> None:
        self.cfg = cfg
        T = cfg.n_steps

        # ── Noise schedule ────────────────────────────────────────────────────
        if cfg.schedule == 'cosine':
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float64)
            f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = betas.clamp(0.0, 0.999)
        else:  # linear
            betas = torch.linspace(cfg.beta_start, cfg.beta_end, T, dtype=torch.float64)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Store as float32 tensors (no device assignment – CPU friendly)
        def r(x: torch.Tensor) -> torch.Tensor:
            return x.float()

        self.betas = r(betas)
        self.alphas_cumprod = r(alphas_cumprod)
        self.alphas_cumprod_prev = r(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = r(alphas_cumprod.sqrt())
        self.sqrt_one_minus_alphas_cumprod = r((1.0 - alphas_cumprod).sqrt())
        self.sqrt_recip_alphas = r((1.0 / alphas).sqrt())
        self.posterior_variance = r(
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Index a 1-D schedule tensor at positions t and broadcast to `shape`."""
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
        """
        Forward diffusion: add noise to x0 at timestep t.

        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

        Parameters
        ----------
        x0 : Tensor, shape (B, C, L)
        t  : LongTensor, shape (B,)
        noise : Tensor, optional – if None, sampled from N(0, I)

        Returns
        -------
        Tensor, shape (B, C, L)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1mb = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mb * noise

    def p_losses(
        self,
        model: UNet1D,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MSE loss for the DDPM noise-prediction objective.

        L = E_t,x0,ε [ || ε - ε_θ(x_t, t) ||² ]
        """
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = model(x_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(
        self,
        model: UNet1D,
        x: torch.Tensor,
        t_idx: int,
    ) -> torch.Tensor:
        """
        One reverse-diffusion step: x_t → x_{t-1}.

        x_{t-1} = (1/√α_t) * (x_t − β_t/√(1−ᾱ_t) * ε_θ(x_t, t))
                  + √β̃_t * z    (z=0 when t==0)
        """
        B = x.shape[0]
        t_tensor = torch.full((B,), t_idx, dtype=torch.long)

        pred_noise = model(x, t_tensor)

        betas_t = self._extract(self.betas, t_tensor, x.shape)
        sqrt_1mb_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t_tensor, x.shape)

        mean = sqrt_recip_alpha_t * (x - betas_t / sqrt_1mb_t * pred_noise)

        if t_idx == 0:
            return mean
        post_var = self._extract(self.posterior_variance, t_tensor, x.shape)
        return mean + post_var.sqrt() * torch.randn_like(x)

    @torch.no_grad()
    def denoise(
        self,
        model: UNet1D,
        x_noisy: torch.Tensor,
        steps: int = 50,
    ) -> torch.Tensor:
        """
        Denoise a real noisy EEG signal using the reverse diffusion chain.

        Strategy
        --------
        Treat ``x_noisy`` as the signal at the highest requested timestep
        ``t_start = steps − 1``.  We add the appropriate amount of DDPM
        diffusion noise to push ``x_noisy`` towards the noise prior, then run
        the reverse chain from ``t_start`` down to ``0``.

        Parameters
        ----------
        model   : trained UNet1D
        x_noisy : Tensor, shape (B, 1, 750)
        steps   : int – number of reverse steps (10 / 25 / 50 etc.)

        Returns
        -------
        Tensor, shape (B, 1, 750) – denoised signal
        """
        model.eval()
        t_start = steps - 1

        # Project x_noisy into the diffusion prior at t_start
        t_tensor = torch.full((x_noisy.shape[0],), t_start, dtype=torch.long)
        noise = torch.randn_like(x_noisy)
        x = self.q_sample(x_noisy, t_tensor, noise)

        # Reverse chain
        for t_idx in reversed(range(steps)):
            x = self.p_sample(model, x, t_idx)

        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_diffusion(
    model: UNet1D,
    diffusion: GaussianDiffusion,
    train_loader,
    val_loader,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    checkpoint_dir: str = 'models/diffusion_teacher',
    save_every: int = 10,
    early_stop_patience: int = 20,
    device: str = 'cpu',
) -> dict:
    """
    Train the UNet1D noise-prediction model.

    Parameters
    ----------
    model          : UNet1D
    diffusion      : GaussianDiffusion
    train_loader   : DataLoader yielding (noisy, clean) pairs
    val_loader     : DataLoader yielding (noisy, clean) pairs
    epochs         : int – number of training epochs
    lr             : float – Adam learning rate
    checkpoint_dir : str – directory for saved checkpoints
    save_every     : int – epoch interval between checkpoints
    early_stop_patience : int – stop if val loss does not improve for this many epochs
    device         : str – 'cpu'

    Returns
    -------
    dict with 'train_losses', 'val_losses', 'best_epoch'
    """
    import os, logging
    logger = logging.getLogger(__name__)

    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, epochs + 1):
        # ── Training ────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for batch_idx, (x_noisy, x_clean) in enumerate(train_loader):
            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)

            # Sample random timesteps for each item in the batch
            t = torch.randint(0, diffusion.cfg.n_steps, (x_clean.shape[0],),
                              device=device, dtype=torch.long)

            optimizer.zero_grad()
            loss = diffusion.p_losses(model, x_clean, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / max(len(train_loader), 1)
        train_losses.append(avg_train)

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_noisy, x_clean in val_loader:
                x_noisy = x_noisy.to(device)
                x_clean = x_clean.to(device)
                t = torch.randint(0, diffusion.cfg.n_steps, (x_clean.shape[0],),
                                  device=device, dtype=torch.long)
                val_loss += diffusion.p_losses(model, x_clean, t).item()
        avg_val = val_loss / max(len(val_loader), 1)
        val_losses.append(avg_val)

        scheduler.step()

        logger.info(
            "Epoch %3d/%d  train_loss=%.5f  val_loss=%.5f  lr=%.2e",
            epoch, epochs, avg_train, avg_val,
            optimizer.param_groups[0]['lr'],
        )

        # ── Checkpointing ────────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'val_loss': avg_val,
                    'cfg': diffusion.cfg,
                },
                os.path.join(checkpoint_dir, 'best_model.pt'),
            )
        else:
            patience_counter += 1

        if epoch % save_every == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'val_loss': avg_val,
                    'cfg': diffusion.cfg,
                },
                os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch:04d}.pt'),
            )

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_counter >= early_stop_patience:
            logger.info("Early stopping at epoch %d (patience=%d).",
                        epoch, early_stop_patience)
            break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
    }
