"""
app.py – Lightweight Flask inference API for the Edge-BCI Distilled Diffusion project.

Endpoints
---------
GET  /              Health check + model inventory
GET  /models        List all available ONNX models with spec cards
POST /denoise       Run inference on a 750-sample EEG signal

Deploy
------
Render:  set Build Command → pip install -r requirements-api.txt
         set Start Command  → gunicorn app:app --bind 0.0.0.0:$PORT
Docker:  docker build -t edge-bci . && docker run -p 5000:5000 edge-bci
Local:   python app.py
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Optional: onnxruntime — graceful error if not installed
# ---------------------------------------------------------------------------
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# Render / Docker: paths are relative to the project root.
# ---------------------------------------------------------------------------
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models/onnx"))

MODEL_SPECS: Dict[str, dict] = {
    "cnn": {
        "display_name": "CNN Student",
        "file": "cnn_student.onnx",
        "params": "~85 K",
        "target_latency_cpu_ms": 20,
        "description": "Balanced edge denoiser. Best default choice.",
    },
    "autoencoder": {
        "display_name": "Autoencoder Student",
        "file": "autoencoder_student.onnx",
        "params": "~45 K",
        "target_latency_cpu_ms": 15,
        "description": "Smallest memory footprint. Suited for ARM / RPi.",
    },
    "consistency": {
        "display_name": "Consistency Student",
        "file": "consistency_student.onnx",
        "params": "~160 K",
        "target_latency_cpu_ms": 10,
        "description": "Single-step consistency distillation. Best one-pass quality.",
    },
}

# Lazy-loaded ONNX sessions — populated on first request per model.
_sessions: Dict[str, Optional["ort.InferenceSession"]] = {}


def _load_session(model_key: str) -> Optional["ort.InferenceSession"]:
    """Load (or return cached) ONNX InferenceSession for *model_key*."""
    if not _ORT_AVAILABLE:
        return None

    if model_key in _sessions:
        return _sessions[model_key]

    spec = MODEL_SPECS.get(model_key)
    if spec is None:
        logger.warning("Unknown model key: %s", model_key)
        _sessions[model_key] = None
        return None

    model_path = MODELS_DIR / spec["file"]
    if not model_path.exists():
        logger.warning("ONNX file not found: %s", model_path)
        _sessions[model_key] = None
        return None

    try:
        sess = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("Loaded ONNX session for '%s' from %s", model_key, model_path)
        _sessions[model_key] = sess
    except Exception as exc:
        logger.error("Failed to load ONNX session for '%s': %s", model_key, exc)
        _sessions[model_key] = None

    return _sessions[model_key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_snr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Signal-to-noise ratio of the denoised signal relative to clean reference."""
    signal_power = float(np.mean(clean ** 2))
    noise_power = float(np.mean((clean - denoised) ** 2))
    if noise_power < 1e-12:
        return 99.0
    return float(10.0 * np.log10(signal_power / noise_power))


def _add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise to *signal* at the requested SNR level."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.randn(*signal.shape).astype(np.float32) * np.sqrt(noise_power)
    return signal + noise


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    """Health check — lists which models are currently available on disk."""
    available = {}
    for key, spec in MODEL_SPECS.items():
        path = MODELS_DIR / spec["file"]
        available[key] = {
            "display_name": spec["display_name"],
            "available": path.exists(),
            "params": spec["params"],
        }

    return jsonify({
        "status": "healthy",
        "service": "Edge-BCI Distilled Diffusion Inference API",
        "onnxruntime_available": _ORT_AVAILABLE,
        "models": available,
    })


@app.route("/models", methods=["GET"])
def list_models():
    """Return full spec cards for all registered models."""
    cards = []
    for key, spec in MODEL_SPECS.items():
        path = MODELS_DIR / spec["file"]
        cards.append({
            "key": key,
            "display_name": spec["display_name"],
            "params": spec["params"],
            "target_latency_cpu_ms": spec["target_latency_cpu_ms"],
            "description": spec["description"],
            "file": spec["file"],
            "available": path.exists(),
        })
    return jsonify({"models": cards})


@app.route("/denoise", methods=["POST"])
def denoise():
    """
    POST /denoise

    Request body (JSON)
    -------------------
    {
        "signal": [<750 float values>],   // required — noisy EEG, one channel
        "model":  "cnn" | "autoencoder" | "consistency",  // optional, default "cnn"
        "snr_db": 10.0                    // optional — for synthetic test mode only
    }

    Response (JSON)
    ---------------
    {
        "denoised": [<750 float values>],
        "latency_ms": 14.7,
        "model_used": "cnn",
        "input_length": 750
    }
    """
    if not _ORT_AVAILABLE:
        return jsonify({"error": "onnxruntime is not installed on this server."}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    # ── Parse signal ──────────────────────────────────────────────────────────
    raw_signal = data.get("signal")
    if raw_signal is None:
        return jsonify({"error": "Missing required field: 'signal'"}), 400

    try:
        signal_np = np.array(raw_signal, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Cannot parse 'signal' as float array: {exc}"}), 400

    if signal_np.ndim != 1 or signal_np.size != 750:
        return jsonify({
            "error": f"'signal' must be a flat array of exactly 750 values, got shape {signal_np.shape}"
        }), 400

    # ── Select model ──────────────────────────────────────────────────────────
    model_key = str(data.get("model", "cnn")).lower()
    if model_key not in MODEL_SPECS:
        return jsonify({
            "error": f"Unknown model '{model_key}'. Valid options: {list(MODEL_SPECS.keys())}"
        }), 400

    sess = _load_session(model_key)
    if sess is None:
        return jsonify({
            "error": f"Model '{model_key}' is not available on this server. "
                     "Check that the .onnx and .onnx.data files are present in models/onnx/."
        }), 503

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        x = signal_np.reshape(1, 1, 750)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        t0 = time.perf_counter()
        result = sess.run([output_name], {input_name: x})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        denoised = np.array(result[0]).reshape(750)

        logger.info("Denoised with '%s' in %.2f ms", model_key, elapsed_ms)
        return jsonify({
            "denoised": denoised.tolist(),
            "latency_ms": round(elapsed_ms, 3),
            "model_used": model_key,
            "input_length": 750,
        })

    except Exception as exc:
        logger.exception("Inference failed for model '%s'", model_key)
        return jsonify({"error": f"Inference error: {exc}"}), 500


# ---------------------------------------------------------------------------
# Entry point (local dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Edge-BCI API on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
