import math

import numpy as np
import torch

from src.data_pipeline import ChannelNormalizer, EEGDataset, inject_noise
from src.diffusion import DiffusionConfig, GaussianDiffusion, UNet1D
from src.distillation import AutoencoderStudent, CNNStudent, ConsistencyStudent
from src.metrics import compute_all_signal_metrics


def test_student_models_preserve_shape_for_batch_sizes() -> None:
    torch.manual_seed(0)
    for batch_size in (1, 2):
        x = torch.randn(batch_size, 1, 32)
        models = [
            CNNStudent(signal_length=32),
            AutoencoderStudent(signal_length=32, latent_dim=8),
            ConsistencyStudent(signal_length=32),
        ]
        for model in models:
            y = model(x)
            assert y.shape == x.shape
            assert torch.isfinite(y).all()


def test_small_diffusion_forward_and_loss_are_finite() -> None:
    torch.manual_seed(0)
    cfg = DiffusionConfig(
        n_steps=8,
        signal_length=32,
        model_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        dropout=0.0,
        time_emb_dim=16,
    )
    model = UNet1D(cfg)
    diffusion = GaussianDiffusion(cfg)

    x = torch.randn(2, 1, 32)
    t = torch.tensor([0, 7], dtype=torch.long)
    out = model(x, t)
    loss = diffusion.p_losses(model, x, t)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(loss)


def test_signal_metrics_report_positive_improvement_for_perfect_denoising() -> None:
    clean = np.ones((2, 1, 32), dtype=np.float32)
    noisy = clean + 0.1
    denoised = clean.copy()

    metrics = compute_all_signal_metrics(clean, noisy, denoised)

    assert metrics["snr_improvement_db"] > 0
    assert metrics["mse"] == 0.0
    assert math.isfinite(metrics["pearson_correlation"])


def test_data_helpers_sanitize_nan_and_inf_values() -> None:
    eeg = np.ones((2, 2, 32), dtype=np.float32)
    eeg[0, 0, 0] = np.nan
    eeg[1, 1, 1] = np.inf
    labels = np.array([1, 2], dtype=np.int64)

    normalizer = ChannelNormalizer().fit(eeg)
    transformed = normalizer.transform(eeg)
    noisy = inject_noise(eeg, snr_db=15.0, rng=np.random.default_rng(0))
    dataset = EEGDataset(eeg, labels, snr_db=15.0, rng_seed=0)
    sample_noisy, sample_clean, sample_label = dataset[0]

    assert np.isfinite(transformed).all()
    assert np.isfinite(noisy).all()
    assert torch.isfinite(sample_noisy).all()
    assert torch.isfinite(sample_clean).all()
    assert sample_label.item() == 0
