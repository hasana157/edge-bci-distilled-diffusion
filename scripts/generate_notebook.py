import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# 1. Setup / Installs
cell_setup = nbf.v4.new_code_cell("""# Install dependencies
!pip install -q torch numpy scipy mne moabb pywavelets matplotlib""")
nb.cells.append(cell_setup)

# 2. Imports and Setup
cell_imports = nbf.v4.new_code_cell("""import os
import sys
import time
import math
import random
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import scipy.signal
import pywt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import mne
import moabb
from moabb.datasets import BCICompetition4_set2a
from moabb.paradigms import MotorImagery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set reproducible seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
""")
nb.cells.append(cell_imports)

# 3. Data Pipeline
with open('src/data_pipeline.py', 'r', encoding='utf-8') as f:
    data_code = f.read()
    # Remove imports that we already have
    lines = [l for l in data_code.split('\n') if not l.startswith('import ') and not l.startswith('from ')]
    cell_data = nbf.v4.new_code_cell('\n'.join(lines))
    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Data Pipeline"))
    nb.cells.append(cell_data)

# 4. Metrics
with open('src/metrics.py', 'r', encoding='utf-8') as f:
    metrics_code = f.read()
    lines = [l for l in metrics_code.split('\n') if not l.startswith('import ') and not l.startswith('from ')]
    cell_metrics = nbf.v4.new_code_cell('\n'.join(lines))
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Metrics"))
    nb.cells.append(cell_metrics)

# 5. Baselines
with open('src/baselines.py', 'r', encoding='utf-8') as f:
    base_code = f.read()
    lines = [l for l in base_code.split('\n') if not l.startswith('import ') and not l.startswith('from ')]
    cell_base = nbf.v4.new_code_cell('\n'.join(lines))
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Baselines (Butterworth, Wavelet, Wiener)"))
    nb.cells.append(cell_base)

# 6. Diffusion Model
with open('src/diffusion.py', 'r', encoding='utf-8') as f:
    diff_code = f.read()
    lines = [l for l in diff_code.split('\n') if not l.startswith('import ') and not l.startswith('from ') and '__future__' not in l]
    cell_diff = nbf.v4.new_code_cell('\n'.join(lines))
    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Diffusion Model (UNet & GaussianDiffusion)"))
    nb.cells.append(cell_diff)

# 7. Training Cell (Adapted from train_diffusion.py)
train_cell = nbf.v4.new_code_cell("""# Configuration for Training
SUBJECTS = [1, 2] # Use [1] for quick test, [1, 2, ..., 9] for full run
EPOCHS = 30       # Set to 100 for full training
BATCH_SIZE = 16
LR = 1e-3
SNR_DB = 10
N_STEPS = 1000

print("=== Phase 3: Training Diffusion Teacher ===")
data = load_bci_competition_data(subjects=SUBJECTS, cache_dir='./data')
X_raw = data['X']
y = data['y']
subject_ids = data['subject']

X_proc = preprocess_pipeline(X_raw, fs=data['fs'])
X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_split(X_proc, y, subject_ids, seed=SEED)

noise_dict_train = inject_noise(X_train, snr_db_list=[SNR_DB])
noise_dict_val   = inject_noise(X_val,   snr_db_list=[SNR_DB])

X_train_noisy = noise_dict_train[f'noisy_{SNR_DB}']
X_val_noisy   = noise_dict_val[f'noisy_{SNR_DB}']

def build_channel_dataset(X_clean, X_noisy, batch_size, shuffle=True):
    n_trials, n_ch, n_samp = X_clean.shape
    clean_flat = X_clean.reshape(n_trials * n_ch, 1, n_samp).astype(np.float32)
    noisy_flat = X_noisy.reshape(n_trials * n_ch, 1, n_samp).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(noisy_flat), torch.from_numpy(clean_flat))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = build_channel_dataset(X_train, X_train_noisy, BATCH_SIZE, True)
val_loader   = build_channel_dataset(X_val, X_val_noisy, BATCH_SIZE, False)

cfg = DiffusionConfig(
    n_steps=N_STEPS, schedule='linear', signal_length=750, in_channels=1, 
    model_channels=32, channel_mult=(1, 2, 4), num_res_blocks=2, dropout=0.0, time_emb_dim=128
)
diffusion = GaussianDiffusion(cfg)
model = UNet1D(cfg)
print(f"Model parameters: {model.num_parameters/1000:.1f}K")

# Set device to GPU if available (Colab)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

history = train_diffusion(
    model, diffusion, train_loader, val_loader,
    epochs=EPOCHS, lr=LR, checkpoint_dir='models/diffusion_teacher',
    save_every=10, early_stop_patience=20, device=device
)

# Plot training curve
plt.figure(figsize=(8, 4))
plt.plot(history['train_losses'], label='Train loss')
plt.plot(history['val_losses'], label='Val loss')
plt.axvline(history['best_epoch'] - 1, color='red', linestyle='--', label=f"Best: {history['best_epoch']}")
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Diffusion Teacher - Training Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
""")
nb.cells.append(nbf.v4.new_markdown_cell("## 5. Train Diffusion Model"))
nb.cells.append(train_cell)

# 8. Evaluation Cell
eval_cell = nbf.v4.new_code_cell("""# ── 1. Evaluate Denoising ──
N_TRIALS = 50        
STEP_COUNTS = [10, 25, 50]

X_test_clean = X_test[:N_TRIALS]
noisy_dict_test = inject_noise(X_test_clean, snr_db_list=[SNR_DB])
X_test_noisy = noisy_dict_test[f'noisy_{SNR_DB}']

def channel_snr(clean, denoised):
    vals = [compute_snr(clean[t, c], denoised[t, c]) for t in range(clean.shape[0]) for c in range(clean.shape[1])]
    vals = [v for v in vals if np.isfinite(v)]
    return np.mean(vals) if vals else float('nan')

snr_noisy = channel_snr(X_test_clean, X_test_noisy)
print(f"Baseline (no denoising) SNR = {snr_noisy:.2f} dB")

model.eval()
model = model.to('cpu') # Evaluate on CPU for fair latency comparison
denoised_results = {}

for steps in STEP_COUNTS:
    print(f"\\nDenoising with {steps} steps...")
    n_trials, n_ch, n_samp = X_test_noisy.shape
    x_flat = torch.from_numpy(X_test_noisy.reshape(n_trials * n_ch, 1, n_samp).astype(np.float32))
    
    with torch.no_grad():
        x_out = diffusion.denoise(model, x_flat, steps=steps)
    X_denoised = x_out.numpy().reshape(n_trials, n_ch, n_samp)
    denoised_results[steps] = X_denoised
    
    snr_val = channel_snr(X_test_clean, X_denoised)
    print(f"Steps={steps} | SNR={snr_val:.2f} dB | Improvement={snr_val - snr_noisy:+.2f} dB")

# ── 2. Plot Comparison ──
t_axis = np.arange(750) / 250.0
trial, ch = 0, 0

n_plots = 2 + len(STEP_COUNTS)
fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.8 * n_plots))
fig.suptitle(f'Diffusion Denoising - Trial {trial}, Channel {ch}', fontsize=12)

axes[0].plot(t_axis, X_test_clean[trial, ch], lw=0.8, color='steelblue')
axes[0].set_title('Clean (ground truth)')

axes[1].plot(t_axis, X_test_noisy[trial, ch], lw=0.8, color='tomato')
axes[1].set_title(f'Noisy (injected @ {SNR_DB} dB)')

colors = ['#2ecc71', '#27ae60', '#1e8449']
for i, steps in enumerate(STEP_COUNTS):
    axes[i + 2].plot(t_axis, denoised_results[steps][trial, ch], lw=0.8, color=colors[i])
    axes[i + 2].set_title(f'Diffusion {steps} steps')

for ax in axes: ax.set_ylabel('Amplitude')
axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()
""")
nb.cells.append(nbf.v4.new_markdown_cell("## 6. Evaluation and Visualization"))
nb.cells.append(eval_cell)

with open('Colab_BCI_Diffusion.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
