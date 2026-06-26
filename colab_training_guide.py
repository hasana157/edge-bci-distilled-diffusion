# EEG BCI Distillation – Colab GPU Training Notebook
# Generated from run_all_experiments.py
# Run on: Google Colab (T4 GPU recommended)
# ============================================================
# CELL 1: Setup environment
# ============================================================
"""
!pip install -q torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
!pip install -q scipy PyWavelets matplotlib psutil onnx onnxruntime
!pip install -q mne  # optional: for advanced EEG loading
"""

# ============================================================
# CELL 2: Clone / mount the repo
# ============================================================
"""
# Option A: mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/ebc

# Option B: clone from GitHub
!git clone https://github.com/hasana157/edge-bci-distilled-diffusion.git /content/ebc
%cd /content/ebc
"""

# ============================================================
# CELL 3: Verify GPU
# ============================================================
"""
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("GPU memory:", torch.cuda.get_device_properties(0).total_memory // 1e9, "GB")
"""

# ============================================================
# CELL 4: Download BCI Competition IV 2a dataset
# ============================================================
"""
# Download from the official BNCI Horizon 2020 mirror
import os, urllib.request

os.makedirs("data/raw", exist_ok=True)
BASE = "https://bnci-horizon-2020.eu/database/data-sets/001-2014/"

subjects = [f"A0{i}T.mat" for i in range(1, 10)]  # A01T.mat ... A09T.mat
for s in subjects:
    url = BASE + s
    dest = f"data/raw/{s}"
    if not os.path.exists(dest):
        print(f"Downloading {s} ...", end=" ")
        urllib.request.urlretrieve(url, dest)
        print("done")
    else:
        print(f"{s} already downloaded")

print("All files:", os.listdir("data/raw"))
"""

# ============================================================
# CELL 5: Run all experiments (full pipeline)
# ============================================================
"""
!python run_all_experiments.py \
    --device cuda \
    --epochs-diffusion 500 \
    --epochs-distill 500 \
    --epochs-classifier 100 \
    --batch-size 64 \
    --data-dir data/raw \
    --seed 42
"""

# ============================================================
# CELL 6: View results
# ============================================================
"""
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image

# Benchmark table
df = pd.read_csv("results/benchmark_results.csv")
print(df[["model", "mean_ms", "p95_ms", "peak_ram_mb"]].to_string(index=False))

# Show plots
for plot in [
    "results/plots/quality_latency_curve.png",
    "results/plots/latency_comparison.png",
    "results/plots/denoising_impact.png",
    "results/plots/confusion_matrix_classifier.png",
]:
    if os.path.exists(plot):
        display(Image(plot))
"""

# ============================================================
# CELL 7: Quick demo (individual student inference)
# ============================================================
"""
import sys, torch
sys.path.insert(0, "src")

from distillation import CNNStudent
from data_pipeline import build_dataloaders

# Load trained student
student = CNNStudent(750)
ckpt = torch.load("models/distilled/cnn_student/best_model.pt", map_location="cuda")
student.load_state_dict(ckpt["model_state"])
student = student.cuda().eval()

# Benchmark latency
loaders = build_dataloaders(dataset_dir="data/raw", batch_size=1)
x_noisy, x_clean, _ = next(iter(loaders["test"]))
x_noisy = x_noisy.cuda()

import time
times = []
for _ in range(100):
    t = time.perf_counter()
    with torch.no_grad():
        _ = student(x_noisy)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t) * 1000)

import numpy as np
print(f"CNN Student GPU latency: {np.mean(times):.2f} ± {np.std(times):.2f} ms (p95={np.percentile(times,95):.2f}ms)")
"""

print("Colab notebook scaffold generated. Copy cells to a Colab notebook.")
