# Architecture

EBC is organized as a small research pipeline with clear boundaries between data, models, evaluation, and deployment.

## Data Flow

```text
BCI Competition IV 2a .mat files
        |
        v
data_pipeline.py
  - load subject files or synthetic fallback
  - trim/pad to 22 channels x 750 samples
  - subject-level train/val/test split
  - train-fitted channel normalization
  - optional Gaussian noise injection
        |
        v
baselines.py / diffusion.py / distillation.py
        |
        v
benchmarking.py + metrics.py
        |
        v
classifier.py closed-loop simulation
        |
        v
results/*.csv and results/plots/*.png
```

## Core Modules

| Module | Responsibility |
|---|---|
| `src/config.py` | Dataclass defaults for data, diffusion, distillation, benchmarking, and closed-loop validation. |
| `src/data_pipeline.py` | Dataset loading, synthetic fallback, preprocessing, subject-safe splits, dataloaders. |
| `src/baselines.py` | Classical Butterworth, wavelet, and Wiener denoising baselines. |
| `src/diffusion.py` | DDPM-style 1-D U-Net teacher and Gaussian diffusion process. |
| `src/distillation.py` | CNN, autoencoder, and consistency students; distillation loss; ONNX export. |
| `src/benchmarking.py` | Latency, memory, throughput measurement and quality-latency plots. |
| `src/classifier.py` | Motor-imagery classifier and closed-loop BCI simulator. |
| `src/metrics.py` | Signal quality and classification metrics. |

## Model Shapes

- Diffusion and student denoisers operate on single-channel EEG windows shaped `(batch, 1, 750)`.
- The classifier operates on multi-channel trials shaped `(batch, 22, 750)`.
- The closed-loop simulator denoises each channel independently before classification.

## Checkpoint Layout

See `models/README.md`. The expected structure is:

```text
models/
|-- diffusion_teacher/
|-- distilled/
|   |-- cnn_student/
|   |-- autoencoder_student/
|   `-- consistency_student/
|-- onnx/
`-- classifier/
```

Large model artifacts are intentionally ignored by Git.
