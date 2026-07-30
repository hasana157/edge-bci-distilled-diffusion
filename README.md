# EBC: Edge-BCI Distilled Diffusion

> **Ultra-fast distilled generative diffusion models for real-time EEG denoising on edge BCI systems.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 🚀 Live Demos

| Platform | Deployment | Link |
|---|---|---|
| **Vercel** | **WebAssembly Dashboard** (In-Browser Inference) | [Try Live Dashboard](https://edge-bci-distilled-diffusion-314trvjn4-hasana-zahid.vercel.app/) |
| **Render** | **REST Inference API** (`POST /denoise`) | [Access API Endpoint](https://edge-bci-inference.onrender.com) |
| **Hugging Face** | **Gradio Interactive Demo** | [Try HF Space](https://huggingface.co/spaces/hasana157/edge-bci-denoising) |

> 🔒 **Privacy First:** On the Vercel dashboard, ONNX model inference runs 100% inside your browser via WebAssembly — no EEG signal data ever leaves your device.

---

## ⚡ Highlights & Benchmarks

Standard diffusion models are accurate for EEG denoising but too slow for closed-loop BCI systems. This project compresses heavy 1-D U-Net diffusion models into ultra-lightweight student denoisers (`<100K` parameters).

| Model | Size | CPU Latency | SNR Gain | Primary Use Case |
|---|---|---:|---:|---|
| **CNN Student** | ~85 K params | **~1.7 ms** | **+12.4 dB** | Real-time edge BCI denoiser |
| **Autoencoder Student** | ~45 K params | **~1.2 ms** | **+9.8 dB** | Ultra-low memory / ARM microcontrollers |
| **Consistency Student** | ~160 K params | **~2.1 ms** | **+14.1 dB** | High-fidelity single-step neural denoising |

---

## 🛠️ Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/hasana157/edge-bci-distilled-diffusion.git
cd edge-bci-distilled-diffusion

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run Edge Inference Benchmark
```bash
python -m src.edge_inference --iterations 100
```

### 3. Run Web Dashboard Locally
```bash
python -m http.server 3000 --directory dashboard
```
Open `http://localhost:3000` in your browser.

---

## 📂 Project Structure

```text
edge-bci-distilled-diffusion/
├── dashboard/              # WebAssembly live dashboard (Vercel)
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── models/onnx/        # Pre-trained ONNX student models
├── src/                    # Core Python engine & pipelines
│   ├── diffusion.py        # 1-D U-Net DDPM teacher model
│   ├── distillation.py     # Knowledge & consistency distillation
│   ├── edge_inference.py   # Lightweight ONNX edge denoiser class
│   └── data_pipeline.py    # EEG preprocessing & synthetic generator
├── app.py                  # Flask REST API server (Render / Docker)
├── Dockerfile              # Production Docker image configuration
└── requirements.txt
```

---

## 📜 License

Released under the [MIT License](LICENSE).
