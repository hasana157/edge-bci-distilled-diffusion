# EBC: Edge BCI Distilled Diffusion

Ultra-fast distilled generative models for real-time EEG denoising on edge BCI systems.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/hasana157/edge-bci-distilled-diffusion/actions/workflows/ci.yml/badge.svg)](https://github.com/hasana157/edge-bci-distilled-diffusion/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why This Exists

Brain-computer interfaces (BCIs) need low-latency EEG processing. Diffusion models can denoise signals well, but standard iterative sampling is usually too slow for closed-loop interaction on edge hardware. This project compresses a diffusion teacher into lightweight student denoisers so BCI pipelines can trade a small amount of signal fidelity for large latency gains.

The repo includes the full research prototype: data loading for BCI Competition IV 2a, classical denoising baselines, a DDPM-style 1-D U-Net teacher, CNN/autoencoder/consistency students, ONNX export, latency benchmarking, and a closed-loop motor-imagery classifier simulation.

## Deliverables

| Objective | Concrete artifacts | Status |
|---|---|---|
| 1. Latency characterization | [src/benchmarking.py](src/benchmarking.py), [evaluation/latency_benchmark.py](evaluation/latency_benchmark.py), [BENCHMARKING_PROTOCOL.md](BENCHMARKING_PROTOCOL.md), [results/README.md](results/README.md) | Implemented |
| 2. Distillation framework | [src/diffusion.py](src/diffusion.py), [src/distillation.py](src/distillation.py), [models/README.md](models/README.md), [models/MODEL_CARDS.md](models/MODEL_CARDS.md) | Implemented |
| 3. Edge hardware optimization | ONNX export in [src/distillation.py](src/distillation.py), [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md), [configs/reproducibility.yaml](configs/reproducibility.yaml) | Implemented |
| 4. Closed-loop validation | [src/classifier.py](src/classifier.py), [run_all_experiments.py](run_all_experiments.py), closed-loop CSV schema in [results/README.md](results/README.md) | Implemented |

## Quick Start

These commands run without downloading the EEG dataset because the data pipeline has a deterministic synthetic fallback for smoke tests.

```bash
git clone https://github.com/hasana157/edge-bci-distilled-diffusion.git
cd edge-bci-distilled-diffusion

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m pytest
```

Run a lightweight baseline verification:

```bash
python src/verify_baselines.py
```

Run a focused latency smoke benchmark:

```bash
python evaluation/latency_benchmark.py --device cpu --iterations 20 --warmup 5
```

Run the full experiment pipeline after placing BCI Competition IV 2a `.mat` files in `data/raw/`:

```bash
python run_all_experiments.py --device cuda --skip-sweep
```

For CPU-only smoke training, keep epochs intentionally tiny:

```bash
python run_all_experiments.py --device cpu --epochs-diffusion 1 --epochs-distill 1 --epochs-classifier 1 --skip-sweep --batch-size 8
```

## Project Structure

```text
edge-bci-distilled-diffusion/
|-- src/
|   |-- config.py             # Central dataclass configuration
|   |-- data_pipeline.py      # Dataset loading, synthetic fallback, preprocessing
|   |-- baselines.py          # Butterworth, wavelet, and Wiener denoisers
|   |-- diffusion.py          # DDPM-style 1-D U-Net teacher
|   |-- distillation.py       # CNN, autoencoder, consistency students, ONNX export
|   |-- benchmarking.py       # Latency, memory, throughput, plots
|   |-- classifier.py         # Motor-imagery classifier and closed-loop simulator
|   `-- metrics.py            # SNR, MSE, accuracy, kappa helpers
|-- data/
|   |-- raw/README.md         # Dataset download and licensing notes
|   `-- processed/README.md   # Generated cache description
|-- models/
|   |-- README.md             # Checkpoint layout
|   `-- MODEL_CARDS.md        # Model cards and limitations
|-- results/
|   |-- README.md             # Result artifact schema
|   |-- baseline_metrics.csv  # Committed baseline smoke result
|   `-- baseline_comparison.png
|-- configs/
|   `-- reproducibility.yaml
|-- tests/
|   `-- test_smoke.py
|-- evaluation/
|   `-- latency_benchmark.py
|-- ARCHITECTURE.md
|-- METHODOLOGY.md
|-- DEPLOYMENT_GUIDE.md
|-- BENCHMARKING_PROTOCOL.md
|-- run_all_experiments.py
`-- requirements.txt
```

## Current Results

The committed result below is a fast baseline sanity check on Subject 1. It is useful for verifying metric plumbing; final research claims should be reproduced with the full benchmark protocol and trained checkpoints.

| Method | SNR after denoising (dB) | SNR improvement (dB) | Latency per window (ms) |
|---|---:|---:|---:|
| Butterworth | 10.5095 | 1.4059 | 10.4168 |
| Wavelet | 0.6026 | -8.5011 | 1.0915 |
| Wiener | 9.8956 | 0.7919 | 0.2841 |

Target performance for trained distilled students:

| Model | Intended role | Target latency | Expected quality |
|---|---|---:|---|
| Diffusion teacher | High-fidelity reference | 50-500 ms depending on steps/device | Highest denoising quality |
| CNN student | Balanced edge denoiser | <20 ms CPU, <5 ms GPU | Near-teacher SNR |
| Autoencoder student | Smallest footprint | <15 ms CPU, <3 ms GPU | Moderate SNR gain |
| Consistency student | Single-step neural denoising | <10 ms CPU, <2 ms GPU | Best one-pass quality |

## Dataset

This project targets BCI Competition IV Dataset 2a:

- 9 subjects, 22 EEG channels, 4 motor-imagery classes.
- 250 Hz sampling, 3-second windows, 750 samples per trial.
- Raw `.mat` files are not committed. See [data/raw/README.md](data/raw/README.md) for download instructions and citation details.
- If files are absent, synthetic EEG is generated for tests and smoke runs. Synthetic results are not scientific evidence.

## Reproducibility

Key reproducibility controls:

- Pinned runtime dependencies in [requirements.txt](requirements.txt).
- Reproducibility settings and benchmark environment in [configs/reproducibility.yaml](configs/reproducibility.yaml).
- Random seeds are set in the data pipeline, training scripts, and experiment runner.
- Benchmark definitions are documented in [BENCHMARKING_PROTOCOL.md](BENCHMARKING_PROTOCOL.md).
- Large datasets and checkpoints are excluded from Git; use external releases, Drive, or Hugging Face Hub for model distribution.

## Documentation

- [TECHNICAL_REPORT.tex](TECHNICAL_REPORT.tex): IEEE-style technical report source with architecture, methodology, results, dashboard, deployment, and references.
- [ARCHITECTURE.md](ARCHITECTURE.md): module map and data flow.
- [METHODOLOGY.md](METHODOLOGY.md): research design, metrics, and limitations.
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md): ONNX export, quantization, and edge runtime notes.
- [BENCHMARKING_PROTOCOL.md](BENCHMARKING_PROTOCOL.md): latency and quality measurement rules.
- [FAQ.md](FAQ.md): common setup and interpretation questions.

## Citation

```bibtex
@misc{ebc2026,
  title        = {EBC: Edge BCI Distilled Diffusion},
  author       = {Hasan},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/hasana157/edge-bci-distilled-diffusion}},
  note         = {Distilled diffusion models for real-time edge BCI EEG denoising}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
