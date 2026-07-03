# Methodology

## Research Question

Can a high-quality diffusion denoiser be distilled into a lightweight neural model that preserves enough EEG signal quality for closed-loop BCI use while meeting edge-device latency targets?

## Dataset

The target dataset is BCI Competition IV 2a:

- 9 subjects.
- 22 EEG channels.
- 4 motor-imagery classes.
- 250 Hz sampling rate.
- 3-second analysis windows, or 750 samples.

The pipeline uses a subject-level split to reduce train/test leakage. Synthetic fallback data exists only to keep tests and examples runnable without the real dataset.

## Denoising Methods

Classical baselines:

- Butterworth bandpass, 4-40 Hz.
- Wavelet denoising with soft-thresholding.
- Frequency-domain Wiener filtering.

Neural models:

- A DDPM-style 1-D U-Net teacher predicts noise for reverse diffusion.
- A CNN student learns one-pass denoising from teacher outputs and clean targets.
- An autoencoder student targets smaller memory footprint.
- A consistency student targets single-step denoising quality.

## Distillation Objective

The student loss combines a soft teacher term and a hard clean-signal term:

```text
loss = alpha * KL(student / T, teacher / T) * T^2 + (1 - alpha) * MSE(student, clean)
```

The default configuration uses `T = 4.0` and `alpha = 0.75`.

## Metrics

Signal quality:

- SNR improvement in dB: `SNR(denoised) - SNR(noisy)`.
- MSE and RMSE against the clean reference.
- Pearson correlation.

Runtime:

- Mean, standard deviation, min, max, p95 latency.
- Throughput in samples per second.
- RAM and VRAM where available.

Closed-loop validation:

- Motor-imagery classification accuracy.
- Cohen kappa.
- End-to-end latency and artifact rejection rate.

## Limitations

- Synthetic fallback data is not scientifically meaningful.
- Final edge claims require benchmark runs on the target hardware.
- The current repository does not commit large pretrained weights.
- EEG generalization should be evaluated across subjects and random seeds before clinical or operational use.
