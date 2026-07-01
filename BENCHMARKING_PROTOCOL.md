# Latency-Fidelity Benchmarking Protocol
### Edge-BCI Distilled Diffusion — EEG Denoising Benchmark Standard

---

## 1. Overview
This document defines the standardized benchmarking protocol used to evaluate and compare
EEG denoising models in this project. It is designed to be reproducible and applicable to
future generative EEG architectures.

---

## 2. Hardware Environment
| Component | Specification |
|---|---|
| Training GPU | NVIDIA T4 (Google Colab) |
| Edge Simulation | CPU-only inference via ONNX Runtime |
| RAM | 12 GB (Colab standard) |
| OS | Linux (Ubuntu 22.04) |

---

## 3. Dataset
- **Name:** BCI Competition IV Dataset 2a
- **Subjects:** 9 participants (A01–A09)
- **Channels:** 22 EEG channels
- **Trial Length:** 3 seconds @ 250 Hz = 750 samples per trial
- **Classes:** 4 motor imagery tasks (left hand, right hand, feet, tongue)
- **Split:** 80% train / 10% val / 10% test (no subject leakage)
- **Noise Injection:** Additive Gaussian noise at SNR levels of 10, 15, and 20 dB

---

## 4. Evaluation Metrics

### 4.1 Signal Quality
| Metric | Formula | Lower is Better |
|---|---|---|
| MSE | Mean Squared Error between denoised and clean signal | ✅ |
| SNR Improvement (dB) | SNR(denoised) − SNR(noisy) | ❌ (higher is better) |
| Val Loss | Combined KL + MSE distillation loss | ✅ |

### 4.2 Inference Latency
- **Input:** Single EEG trial — shape `(1, 1, 750)`, dtype `float32`
- **Warmup Runs:** 10 iterations (discarded)
- **Measurement Runs:** 100 iterations
- **Reported Stats:** Mean (ms), Std (ms), Min (ms), Max (ms), P95 (ms)
- **Runtime:** ONNX Runtime 1.x, `CPUExecutionProvider`

---

## 5. Models Benchmarked
| Model | Parameters | Type |
|---|---|---|
| Butterworth Filter | N/A | Classical DSP |
| Wavelet Denoising | N/A | Classical DSP |
| Wiener Filter | N/A | Classical DSP |
| Diffusion Teacher (UNet1D) | ~15.9M | Deep Generative |
| CNN Student | ~3.3M | Distilled Edge |
| Autoencoder Student | ~45K | Distilled Edge |
| Consistency Student | ~1.2M | Distilled Edge |

---

## 6. Reproducing the Benchmark
```bash
# Clone the repository
git clone https://github.com/hasana157/edge-bci-distilled-diffusion.git
cd edge-bci-distilled-diffusion

# Run full pipeline including benchmarks
python run_all_experiments.py --skip-training --batch-size 128
```
All results are saved to `results/benchmark_results.csv` and plots to `results/plots/`.

---

## 7. Results Reference
See `results/benchmark_results.csv` for full numeric results and
`results/plots/quality_latency_curve.png` for the quality-latency trade-off visualization.
