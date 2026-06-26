# EBC: Edge BCI Distilled Diffusion

## Ultra-Fast Distilled Generative Models for Real-Time Edge-Deployed BCIs
### GPU-Accelerated Internship Version | 2026

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Project Overview

This project implements a **knowledge distillation framework** that compresses large Diffusion Models into lightweight neural networks for **real-time EEG signal denoising** on edge devices.

### Key Contributions
- 🧠 **Diffusion Model Teacher** (2M+ params): DDPM-style UNet1D trained on EEG denoising
- ⚡ **CNN Student** (~85K params): 10–50× faster, <10ms GPU inference
- 📦 **Autoencoder Student** (~45K params): Ultra-lightweight, latent bottleneck
- 🔁 **Consistency Student** (NEW): Single-step denoising (FR-406)
- 🔌 **ONNX Export** (NEW): Edge simulation with CPU inference profiling (FR-350)
- 📊 **Quality-Latency Curves** (NEW): Documented trade-offs (Gap 4)
- 🤖 **Closed-Loop BCI Simulation**: Motor imagery classification pipeline

---

## Repository Structure

```
edge-bci-distilled-diffusion/
│
├── src/                          # Core source modules
│   ├── config.py                 # Central configuration dataclasses
│   ├── data_pipeline.py          # BCI Comp IV 2a loading + preprocessing
│   ├── baselines.py              # Butterworth / Wavelet / Wiener filters
│   ├── diffusion.py              # DDPM UNet1D (teacher model)
│   ├── distillation.py           # CNN / AE / Consistency students + ONNX
│   ├── benchmarking.py           # Latency, memory, CPU utilization
│   ├── classifier.py             # Motor imagery EEGNet classifier
│   └── metrics.py                # SNR, MSE, Kappa, accuracy metrics
│
├── data/
│   ├── raw/                      # BCI Competition IV 2a .mat files (not tracked)
│   │   └── README.md             # Download instructions
│   └── processed/                # Cached preprocessed arrays (generated)
│
├── models/
│   ├── diffusion_teacher/        # Teacher checkpoints (best_model.pt)
│   ├── distilled/
│   │   ├── cnn_student/          # CNN student checkpoints
│   │   ├── autoencoder_student/  # AE student checkpoints
│   │   └── consistency_student/  # Consistency student checkpoints
│   ├── onnx/                     # Exported ONNX models
│   └── classifier/               # Motor imagery classifier
│
├── results/
│   ├── plots/                    # Publication-quality figures
│   │   ├── quality_latency_curve.png
│   │   ├── latency_comparison.png
│   │   ├── denoising_impact.png
│   │   ├── confusion_matrix_classifier.png
│   │   ├── diffusion_training.png
│   │   └── *_training.png
│   ├── benchmark_results.csv     # Main results table
│   └── closed_loop_impact.csv    # Denoising impact on classification
│
├── srs/                          # Software Requirements Specification
│   ├── SRS_BCI_Distillation_Internship.docx
│   └── update srs.txt            # SRS gap review (GPU upgrade notes)
│
├── run_all_experiments.py        # FR-704: Master one-command runner
├── colab_training_guide.py       # Colab cell-by-cell guide
├── requirements.txt              # Pinned dependencies
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

See [`data/raw/README.md`](data/raw/README.md) for download instructions.

Or use the Colab script:
```python
import urllib.request, os
BASE = "https://bnci-horizon-2020.eu/database/data-sets/001-2014/"
os.makedirs("data/raw", exist_ok=True)
for i in range(1, 10):
    fname = f"A0{i}T.mat"
    urllib.request.urlretrieve(BASE + fname, f"data/raw/{fname}")
```

### 3. Run All Experiments

```bash
# Colab T4 GPU (recommended):
python run_all_experiments.py --device cuda

# CPU-only (slow, for testing):
python run_all_experiments.py --device cpu --epochs-diffusion 100

# Skip training, benchmark only:
python run_all_experiments.py --skip-training
```

### 4. Open the Colab Notebook

See `colab_training_guide.py` for cell-by-cell Colab instructions.

---

## SRS Compliance Table (Updated — GPU Version)

| Requirement | ID | Status | Notes |
|---|---|---|---|
| Load BCI Comp IV 2a | FR-101 | ✅ | `data_pipeline.py` |
| 250 Hz standardization | FR-102 | ✅ | `data_pipeline.py` |
| 750-sample windows | FR-103 | ✅ | `data_pipeline.py` |
| Channel z-score norm | FR-104 | ✅ | `ChannelNormalizer` |
| 80/10/10 split | FR-105 | ✅ | Subject-level, no leakage |
| Butterworth 4-40 Hz | FR-201 | ✅ | `baselines.py` |
| Wavelet BayesShrink | FR-202 | ✅ | `baselines.py` |
| Wiener filter | FR-203 | ✅ | `baselines.py` |
| Noise injection | FR-204 | ✅ | 10/15/20 dB SNR |
| Artifact flagging | FR-205 | ✅ | Amplitude + RMS |
| DDPM diffusion model | FR-301 | ✅ | `diffusion.py` |
| Training loop | FR-302 | ✅ | 500 epochs, Colab T4 |
| Variable-step inference | FR-303 | ✅ | 10/25/50/100/500 steps |
| CPU/GPU architecture | FR-304 | ✅ | 2M params default |
| Checkpoint save/load | FR-305 | ✅ | `save_checkpoint()` |
| ONNX export | FR-350 | ✅ NEW | `distillation.py` |
| Knowledge distillation | FR-401 | ✅ | `distillation.py` |
| CNN student | FR-402 | ✅ | ~85K params |
| Autoencoder student | FR-403 | ✅ | ~45K params, latent-64 |
| KL loss (T=4) | FR-404 | ✅ | 75% soft + 25% hard |
| Hyperparameter sweep | FR-405 | ✅ | LR × alpha grid |
| Consistency distillation | FR-406 | ✅ NEW | Single-step student |
| Latency measurement | FR-501 | ✅ | 100 iters, p95 |
| Memory profiling | FR-502 | ✅ | RAM + VRAM |
| CPU utilization | FR-503 | ✅ | psutil |
| Throughput | FR-504 | ✅ | samples/sec |
| Benchmark suite | FR-505 | ✅ | All models/configs |
| MI classifier | FR-601 | ✅ | EEGNet-style CNN |
| End-to-end pipeline | FR-602 | ✅ | <50ms GPU target |
| Denoising impact | FR-603 | ✅ | ∆Accuracy per method |
| Real-time visualization | FR-604 | ✅ | `plot_denoising_impact` |
| Artifact handling | FR-605 | ✅ | Skip + log |
| Results CSV | FR-701 | ✅ | `benchmark_results.csv` |
| Quality-latency curve | FR-702 | ✅ NEW | `quality_latency_curve.png` |
| Technical report | FR-703 | ⬜ | To be written |
| Single-command runner | FR-704 | ✅ | `run_all_experiments.py` |
| GitHub README | FR-705 | ✅ | This file |

---

## Performance Targets (SRS Updated)

| Model | Latency (GPU) | Latency (CPU) | SNR Improvement |
|---|---|---|---|
| Butterworth | ~0.5ms | ~1ms | 2–3 dB |
| Wavelet | ~2ms | ~5ms | 3–4 dB |
| Wiener | ~2ms | ~5ms | 3–3.5 dB |
| Diffusion 50-step | ~50ms | ~500ms | 4–5 dB |
| CNN Student | **<5ms** | **<20ms** | 3.5–4.5 dB |
| AE Student | **<3ms** | **<15ms** | 3–4 dB |
| Consistency (1-step) | **<2ms** | **<10ms** | 3–4 dB |

---

## Requirements

```
torch>=2.0.0
scipy>=1.10.0
PyWavelets>=1.4.0
matplotlib>=3.7.0
numpy>=1.24.0
psutil>=5.9.0
onnx>=1.14.0
onnxruntime>=1.15.0
```

---

## Citation

If you use this code, please cite:
```
@misc{ebc2026,
  title={Ultra-Fast Distilled Generative Models for Real-Time Edge-Deployed BCIs},
  author={Hasan},
  year={2026},
  publisher={GitHub},
  url={https://github.com/hasana157/edge-bci-distilled-diffusion}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
