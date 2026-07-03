# Latency-Fidelity Benchmarking Protocol

This document defines the standard measurement rules for EBC EEG denoising benchmarks.

## Hardware Environment

Report the exact hardware used with every benchmark table:

| Field | Example |
|---|---|
| CPU | Intel i7 laptop CPU |
| GPU | NVIDIA T4 on Google Colab |
| RAM | 12 GB |
| OS | Ubuntu 22.04 |
| Runtime | PyTorch or ONNX Runtime |
| Provider | CPUExecutionProvider, CUDAExecutionProvider, or PyTorch CUDA |

## Dataset

- Dataset: BCI Competition IV Dataset 2a.
- Subjects: 9 participants.
- Channels: 22 EEG channels.
- Window: 3 seconds at 250 Hz, or 750 samples.
- Classes: left hand, right hand, feet, tongue.
- Split: subject-level 80/10/10 where possible.
- Noise: additive Gaussian noise at configured SNR levels, usually 10, 15, and 20 dB.

Synthetic fallback data is allowed only for smoke tests.

## Signal Quality Metrics

| Metric | Definition | Direction |
|---|---|---|
| MSE | Mean squared error between denoised and clean signal | Lower is better |
| RMSE | Root mean squared error | Lower is better |
| SNR improvement | `SNR(denoised) - SNR(noisy)` in dB | Higher is better |
| Pearson correlation | Correlation between clean and denoised flattened signals | Higher is better |

SNR uses mean squared signal and noise power:

```text
SNR = 10 * log10(mean(clean^2) / mean(noise^2))
```

## Latency Metrics

Default settings:

- Input shape: `(1, 1, 750)`.
- Dtype: `float32`.
- Warmup: 10 runs discarded.
- Measurement: 100 runs.
- Report: mean, standard deviation, min, max, p95, throughput.
- CUDA timing: synchronize before and after measured calls.
- CPU timing: use `time.perf_counter()`.

Do not include dataset download, preprocessing cache creation, plotting, or training in inference latency.

## Reproduction Commands

Install dependencies and run smoke tests:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Run baseline verification:

```bash
python src/verify_baselines.py
```

Run a student latency smoke benchmark:

```bash
python evaluation/latency_benchmark.py --device cpu --iterations 20 --warmup 5
```

Run the full pipeline on a GPU:

```bash
python run_all_experiments.py --device cuda --skip-sweep
```

Run a tiny CPU smoke pipeline:

```bash
python run_all_experiments.py --device cpu --epochs-diffusion 1 --epochs-distill 1 --epochs-classifier 1 --skip-sweep --batch-size 8
```

## Required Result Artifacts

- `results/benchmark_results.csv`
- `results/closed_loop_impact.csv`
- `results/plots/latency_comparison.png`
- `results/plots/quality_latency_curve.png`
- `configs/reproducibility.yaml` or equivalent saved config

Every published benchmark should state whether it used real BCI data or synthetic fallback data.
