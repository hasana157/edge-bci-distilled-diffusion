"""
edge_inference.py – Standalone edge inference module for the EBC project.

Wraps ONNX Runtime in a simple class designed for deployment on resource-
constrained hardware (Raspberry Pi 4, NVIDIA Jetson Nano, etc.).

Usage
-----
As a library::

    from src.edge_inference import EdgeBCIDenoiser

    denoiser = EdgeBCIDenoiser("models/onnx/cnn_student.onnx")
    denoised, latency_ms = denoiser.denoise(noisy_750_sample_array)

As a CLI smoke-test::

    python -m src.edge_inference                        # uses CNN student
    python -m src.edge_inference --model autoencoder    # autoencoder student
    python -m src.edge_inference --model consistency    # consistency student
    python -m src.edge_inference --iterations 50        # run 50 iterations
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Default model paths (relative to project root)
# ─────────────────────────────────────────────────────────────────────────────

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "onnx"

_DEFAULT_MODELS = {
    "cnn":          _MODELS_DIR / "cnn_student.onnx",
    "autoencoder":  _MODELS_DIR / "autoencoder_student.onnx",
    "consistency":  _MODELS_DIR / "consistency_student.onnx",
}


# ─────────────────────────────────────────────────────────────────────────────
# EdgeBCIDenoiser
# ─────────────────────────────────────────────────────────────────────────────

class EdgeBCIDenoiser:
    """
    Lightweight ONNX-based EEG denoiser for edge hardware deployment.

    Parameters
    ----------
    model_path : str | Path
        Path to the .onnx model file.  If the model uses external weights
        (a companion .onnx.data file), both files must be in the same directory.
    providers : list[str] | None
        ONNX Runtime execution providers.  Defaults to CPU only, which works
        on all hardware.  Pass ``["CUDAExecutionProvider", "CPUExecutionProvider"]``
        on Jetson / desktop GPU.
    """

    def __init__(
        self,
        model_path: "str | Path",
        providers: Optional[list] = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for edge inference. "
                "Install it with:  pip install onnxruntime"
            ) from exc

        self._model_path = Path(model_path).resolve()
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self._model_path}\n"
                "Run training + export first, or check that the file path is correct."
            )

        providers = providers or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self._model_path), providers=providers)
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        logger.info(
            "EdgeBCIDenoiser ready: model=%s  providers=%s",
            self._model_path.name,
            providers,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def denoise(self, eeg_signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Denoise a single-channel 750-sample EEG window.

        Parameters
        ----------
        eeg_signal : np.ndarray
            Raw EEG signal.  Accepted shapes:
              - ``(750,)``          → flat array
              - ``(1, 750)``        → single channel
              - ``(1, 1, 750)``     → batch=1, ch=1, samples

        Returns
        -------
        denoised : np.ndarray, shape (750,)
            Denoised signal.
        latency_ms : float
            Wall-clock inference time in milliseconds (excluding overhead).
        """
        x = np.asarray(eeg_signal, dtype=np.float32).reshape(1, 1, 750)

        t0 = time.perf_counter()
        output = self._session.run([self._output_name], {self._input_name: x})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        denoised = np.array(output[0]).reshape(750)
        return denoised, latency_ms

    def benchmark(self, iterations: int = 100, warmup: int = 10) -> dict:
        """
        Run a latency benchmark with synthetic input.

        Parameters
        ----------
        iterations : int
            Number of timed inference calls.
        warmup : int
            Number of untimed warmup calls before measurement.

        Returns
        -------
        dict with keys: mean_ms, p50_ms, p95_ms, p99_ms, min_ms, max_ms
        """
        dummy = np.random.randn(750).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            self.denoise(dummy)

        # Timed runs
        latencies = []
        for _ in range(iterations):
            _, ms = self.denoise(dummy)
            latencies.append(ms)

        latencies_arr = np.array(latencies)
        return {
            "mean_ms": float(np.mean(latencies_arr)),
            "p50_ms":  float(np.percentile(latencies_arr, 50)),
            "p95_ms":  float(np.percentile(latencies_arr, 95)),
            "p99_ms":  float(np.percentile(latencies_arr, 99)),
            "min_ms":  float(np.min(latencies_arr)),
            "max_ms":  float(np.max(latencies_arr)),
        }

    @property
    def model_name(self) -> str:
        return self._model_path.stem


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point: python -m src.edge_inference
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edge-BCI denoiser smoke-test and latency benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=list(_DEFAULT_MODELS.keys()),
        default="cnn",
        help="Which student model to benchmark.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override model path (absolute or relative to CWD).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of timed inference iterations.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations before timing.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    # Resolve model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = _DEFAULT_MODELS[args.model]

    # Initialise
    try:
        denoiser = EdgeBCIDenoiser(model_path)
    except (FileNotFoundError, ImportError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Edge-BCI Denoiser  |  Model: {denoiser.model_name}")
    print(f"{'=' * 60}\n")

    # Single inference smoke-test
    dummy_signal = np.random.randn(750).astype(np.float32)
    denoised, first_latency = denoiser.denoise(dummy_signal)
    print(f"  Smoke test         : OK  (first inference: {first_latency:.2f} ms)")
    print(f"  Input  shape       : {dummy_signal.shape}")
    print(f"  Output shape       : {denoised.shape}\n")

    # Full benchmark
    print(f"  Running benchmark  : {args.warmup} warmup + {args.iterations} timed iterations ...")
    stats = denoiser.benchmark(iterations=args.iterations, warmup=args.warmup)
    print()
    print(f"  +---------------------------------+  ")
    print(f"  |  Latency Benchmark Results      |  ")
    print(f"  +---------------------------------+  ")
    print(f"  |  Mean    : {stats['mean_ms']:>8.2f} ms            |  ")
    print(f"  |  p50     : {stats['p50_ms']:>8.2f} ms            |  ")
    print(f"  |  p95     : {stats['p95_ms']:>8.2f} ms            |  ")
    print(f"  |  p99     : {stats['p99_ms']:>8.2f} ms            |  ")
    print(f"  |  Min     : {stats['min_ms']:>8.2f} ms            |  ")
    print(f"  |  Max     : {stats['max_ms']:>8.2f} ms            |  ")
    print(f"  +---------------------------------+  ")
    print()

    # Pass/fail vs edge target
    target_ms = 20.0
    status = "[PASS]" if stats["p95_ms"] < target_ms else "[FAIL]"
    print(f"  p95 vs <{target_ms}ms target : {status}  ({stats['p95_ms']:.2f} ms)\n")


if __name__ == "__main__":
    main()
