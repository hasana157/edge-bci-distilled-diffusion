"""
app.py — Hugging Face Spaces Gradio demo for Edge-BCI Distilled Diffusion.

What it does
------------
* Generates a synthetic 3-second motor-imagery EEG snippet (alpha + beta + drift)
  at the user-selected SNR level.
* Runs the selected knowledge-distilled ONNX student model entirely in-process
  (no network calls, privacy-safe).
* Returns a 3-panel matplotlib figure: noisy input / denoised output / ground-truth
  overlay + latency and SNR-improvement annotations.

The three ONNX models are bundled directly in the Space repository so the demo
works out-of-the-box with zero user interaction.

Upload instructions
-------------------
When creating the Hugging Face Space, also copy:
    models/onnx/cnn_student.onnx
    models/onnx/cnn_student.onnx.data
    models/onnx/autoencoder_student.onnx
    models/onnx/autoencoder_student.onnx.data
    models/onnx/consistency_student.onnx
    models/onnx/consistency_student.onnx.data

into the Space repo under the same `models/onnx/` path.
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

matplotlib.use("Agg")  # headless backend for Spaces

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

_MODELS_DIR = Path(__file__).resolve().parent / "models" / "onnx"

_REGISTRY = {
    "CNN Student (~85 K params)": _MODELS_DIR / "cnn_student.onnx",
    "Autoencoder Student (~45 K params)": _MODELS_DIR / "autoencoder_student.onnx",
    "Consistency Student (~160 K params)": _MODELS_DIR / "consistency_student.onnx",
}

# Pre-load all sessions at startup (avoids cold-start lag on first request)
_SESSIONS: dict[str, ort.InferenceSession] = {}
for name, path in _REGISTRY.items():
    if path.exists():
        _SESSIONS[name] = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
    else:
        print(f"[WARN] Model file not found, skipping: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic EEG generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_eeg(snr_db: float, signal_length: int = 750) -> tuple:
    """
    Generate a clean motor-imagery EEG trace and a noisy version at *snr_db*.

    Returns
    -------
    clean : np.ndarray, shape (signal_length,)
    noisy : np.ndarray, shape (signal_length,)
    """
    fs = 250  # 250 Hz
    t  = np.linspace(0, signal_length / fs, signal_length)

    # Motor imagery components: alpha (10 Hz) + beta (20 Hz) + slow drift (1 Hz)
    clean = (
        np.sin(2 * np.pi * 10 * t) * 0.6   # alpha
        + np.sin(2 * np.pi * 20 * t) * 0.4  # beta
        + np.sin(2 * np.pi *  1 * t) * 0.2  # baseline drift
    ).astype(np.float32)

    signal_power  = float(np.mean(clean ** 2))
    noise_power   = signal_power / (10 ** (snr_db / 10.0))
    noise         = np.random.randn(signal_length).astype(np.float32) * np.sqrt(noise_power)

    return clean, clean + noise


# ─────────────────────────────────────────────────────────────────────────────
# Core inference function (called by Gradio)
# ─────────────────────────────────────────────────────────────────────────────

def denoise_eeg(model_name: str, snr_db: float) -> tuple:
    """
    Generate a noisy EEG, denoise it with the selected model, and return a
    3-panel figure plus a metrics summary string.

    Parameters
    ----------
    model_name : str   — one of the keys in _REGISTRY
    snr_db     : float — input signal-to-noise ratio in dB

    Returns
    -------
    fig     : matplotlib Figure
    summary : str  — key metrics formatted as a short report
    """
    # ── Validate model selection ──────────────────────────────────────────────
    sess = _SESSIONS.get(model_name)
    if sess is None:
        return None, f"❌ Model '{model_name}' is not loaded. Check that the .onnx files are present."

    # ── Generate signal ───────────────────────────────────────────────────────
    clean, noisy = _generate_synthetic_eeg(snr_db=snr_db)

    # ── Run inference ─────────────────────────────────────────────────────────
    x           = noisy.reshape(1, 1, 750)
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    t0       = time.perf_counter()
    result   = sess.run([output_name], {input_name: x})
    latency  = (time.perf_counter() - t0) * 1000.0

    denoised = np.array(result[0]).reshape(750)

    # ── Compute metrics ───────────────────────────────────────────────────────
    mse = float(np.mean((denoised - clean) ** 2))

    # SNR of noisy vs clean
    noise_power_in  = float(np.mean((noisy   - clean) ** 2))
    noise_power_out = float(np.mean((denoised - clean) ** 2))
    signal_power    = float(np.mean(clean ** 2))

    snr_in  = 10 * np.log10(signal_power / (noise_power_in  + 1e-12))
    snr_out = 10 * np.log10(signal_power / (noise_power_out + 1e-12))
    snr_gain = snr_out - snr_in

    # ── 3-panel figure ────────────────────────────────────────────────────────
    time_axis = np.arange(750) / 250.0  # seconds

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), facecolor="#0a0d1a")
    fig.suptitle(
        f"EBC — {model_name}  |  Input SNR: {snr_db:.0f} dB  |  Latency: {latency:.2f} ms",
        color="#f3f4f6", fontsize=13, fontweight="bold", y=0.98,
    )

    _plot_cfg = dict(facecolor="#0e1628")
    panel_data = [
        (axes[0], noisy,    "#ef4444", "Noisy Input",     f"SNR = {snr_in:.1f} dB"),
        (axes[1], denoised, "#10b981", "Denoised Output", f"SNR = {snr_out:.1f} dB  |  MSE = {mse:.4f}"),
        (axes[2], clean,    "#3b82f6", "Ground Truth",    "Reference"),
    ]

    for ax, signal, color, title, subtitle in panel_data:
        ax.set_facecolor("#0e1628")
        ax.plot(time_axis, signal, color=color, linewidth=1.2, alpha=0.9)
        if title == "Denoised Output":
            # Overlay clean reference as dashed guide
            ax.plot(time_axis, clean, color="#3b82f6", linewidth=1.0,
                    linestyle="--", alpha=0.4, label="Clean reference")
        ax.set_title(f"{title} — {subtitle}", color="#d1d5db", fontsize=10, pad=4)
        ax.set_ylabel("Amplitude (a.u.)", color="#9ca3af", fontsize=9)
        ax.tick_params(colors="#6b7280", labelsize=8)
        ax.spines[:].set_color("#1f2937")
        ax.grid(True, color="rgba(255,255,255,0.04)" if False else "#1a2033", linewidth=0.5)

    axes[2].set_xlabel("Time (s)", color="#9ca3af", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # ── Text summary ──────────────────────────────────────────────────────────
    summary = (
        f"**Model:** {model_name}\n"
        f"**Input SNR:** {snr_in:.1f} dB  →  **Output SNR:** {snr_out:.1f} dB  "
        f"(**+{snr_gain:.1f} dB improvement**)\n"
        f"**MSE:** {mse:.4f}\n"
        f"**Inference latency:** {latency:.2f} ms  "
        f"({'✅ sub-20ms' if latency < 20 else '⚠️ above 20ms edge target'})\n\n"
        f"*Inference runs entirely on CPU inside the HF Space container. "
        f"On a browser via Vercel, ONNX Runtime Web executes the same model "
        f"inside WebAssembly in your own device.*"
    )

    return fig, summary


# ─────────────────────────────────────────────────────────────────────────────
# Gradio Interface
# ─────────────────────────────────────────────────────────────────────────────

available_models = list(_SESSIONS.keys()) or list(_REGISTRY.keys())

demo = gr.Interface(
    fn=denoise_eeg,
    inputs=[
        gr.Dropdown(
            choices=available_models,
            value=available_models[0] if available_models else None,
            label="Distilled Student Model",
        ),
        gr.Slider(
            minimum=5, maximum=25, value=10, step=1,
            label="Input SNR (dB) — lower = noisier EEG",
        ),
    ],
    outputs=[
        gr.Plot(label="Denoising Result"),
        gr.Markdown(label="Metrics Summary"),
    ],
    title="EBC: Edge-BCI Distilled Diffusion — Live Demo",
    description=(
        "Real-time EEG denoising via knowledge-distilled DDPM students. "
        "Select a model, set the noise level, and click **Submit** to run inference. "
        "Synthetic motor-imagery EEG is generated on the fly — no data upload required.\n\n"
        "[📦 GitHub](https://github.com/hasana157/edge-bci-distilled-diffusion) | "
        "[📊 Live Dashboard (Vercel)](https://edge-bci.vercel.app)"
    ),
    examples=[
        [available_models[0] if available_models else "CNN Student (~85 K params)", 10],
        [available_models[0] if available_models else "CNN Student (~85 K params)", 5],
        [available_models[-1] if len(available_models) > 1 else available_models[0], 15],
    ],
    allow_flagging="never",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="emerald",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0a0d1a",
        body_text_color="#f3f4f6",
        block_background_fill="#0e1628",
        block_border_color="#1f2937",
    ),
)

if __name__ == "__main__":
    demo.launch()
