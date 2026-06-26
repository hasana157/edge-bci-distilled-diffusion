"""
config.py – Central configuration dataclasses for the EBC project.

GPU-Accelerated Internship Version (updated per SRS gap review).
All defaults target Colab T4.  CPU-fallback values are annotated inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / Data Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """FR-101 to FR-105 – Data pipeline configuration."""

    dataset_dir: str = "data/raw"                 # path to .mat files
    processed_dir: str = "data/processed"
    sampling_rate: int = 250                       # Hz (FR-102)
    signal_length: int = 750                       # samples = 3 s × 250 Hz (FR-103)
    n_channels: int = 22                           # BCI Competition IV 2a (FR-101)
    n_classes: int = 4                             # left/right/feet/tongue

    # Train / val / test split ratios (no subject leakage) – FR-105
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Noise injection SNR levels (dB) – FR-204
    snr_levels: List[int] = field(default_factory=lambda: [10, 15, 20])

    random_seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion Model  (updated: up to 2M–5M params, 500+ epochs – SRS UPGRADE)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiffusionConfig:
    """FR-301 to FR-305 – Diffusion model and noise schedule configuration."""

    # Noise schedule
    n_steps: int = 1000                            # training timesteps
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"                       # 'linear' | 'cosine'

    # Inference step counts to benchmark – SRS UPGRADE (added 100, 500)
    inference_steps: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 500])

    # Signal dimensions
    signal_length: int = 750
    in_channels: int = 1

    # UNet architecture – SRS UPGRADE: allow up to 2M–5M params on Colab
    model_channels: int = 64                       # base feature width
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)  # → 64,128,256,512
    num_res_blocks: int = 2
    dropout: float = 0.1
    time_emb_dim: int = 256

    # Training – SRS UPGRADE: 500+ epochs on Colab
    epochs: int = 500
    batch_size: int = 64                           # Colab T4; reduce to 16 on CPU
    lr: float = 1e-3
    lr_step: int = 100
    lr_gamma: float = 0.5
    grad_clip: float = 1.0
    early_stop_patience: int = 50
    save_every: int = 25
    checkpoint_dir: str = "models/diffusion_teacher"


# ─────────────────────────────────────────────────────────────────────────────
# Distillation  (CNN + Autoencoder students)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DistillationConfig:
    """FR-401 to FR-405 + FR-406 (consistency) – Distillation configuration."""

    # Temperature-scaled KL loss – FR-404
    temperature: float = 4.0
    alpha: float = 0.75                            # soft target weight (0.75 soft + 0.25 hard)

    # Training schedule – SRS UPGRADE: 500+ epochs feasible on Colab
    epochs: int = 500
    batch_size: int = 128
    lr: float = 5e-4
    lr_step: int = 100
    lr_gamma: float = 0.5
    early_stop_patience: int = 50
    save_every: int = 25
    checkpoint_dir: str = "models/distilled"

    # Hyperparameter sweep ranges – FR-405
    lr_sweep: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 1e-2])
    alpha_sweep: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.75, 0.9])

    # FR-406 – Consistency distillation (single-step)
    enable_consistency: bool = True
    consistency_epochs: int = 200
    consistency_lr: float = 2e-4

    # ONNX export – FR-350 (new)
    onnx_export_dir: str = "models/onnx"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarking  (FR-501 to FR-505)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """FR-501 to FR-505 – Benchmarking configuration."""

    n_iterations: int = 100                        # iterations per measurement (FR-501)
    warmup: int = 10                               # warm-up iterations (not counted)

    # GPU latency targets (SRS UPGRADE)
    gpu_latency_target_ms: float = 10.0            # <10 ms on GPU
    cpu_latency_target_ms: float = 50.0            # <50 ms on CPU

    results_dir: str = "results"
    plots_dir: str = "results/plots"


# ─────────────────────────────────────────────────────────────────────────────
# Closed-Loop Simulation  (FR-601 to FR-605)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClosedLoopConfig:
    """FR-601 to FR-605 – Closed-loop BCI simulation configuration."""

    # SRS UPGRADE: ≥80% with GPU-trained models
    accuracy_target: float = 0.80

    # SRS UPGRADE: primary target <50 ms GPU, secondary <500 ms CPU
    latency_target_gpu_ms: float = 50.0
    latency_target_cpu_ms: float = 500.0

    # SNR targets – SRS UPGRADE: 5–8 dB achievable with GPU teacher
    snr_target_min_db: float = 5.0
    snr_target_max_db: float = 8.0

    # Classifier
    classifier_epochs: int = 100
    classifier_lr: float = 1e-3
    classifier_checkpoint_dir: str = "models/classifier"


# ─────────────────────────────────────────────────────────────────────────────
# Master config (aggregate)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Top-level configuration object. Pass this around instead of globals."""

    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    distill: DistillationConfig = field(default_factory=DistillationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    closed_loop: ClosedLoopConfig = field(default_factory=ClosedLoopConfig)

    # Runtime device – auto-detected in main scripts
    device: str = "cuda"                           # override to 'cpu' if needed
