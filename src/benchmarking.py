"""
benchmarking.py – Comprehensive latency and quality benchmarking framework.

FR-501  Latency measurement: 100 iterations, mean/std/min/max/p95.
FR-502  Memory profiling: peak RAM during inference.
FR-503  CPU utilization monitoring (psutil).
FR-504  Throughput: samples/second.
FR-505  Comparative benchmark suite (all methods).

SRS UPGRADE:
  - GPU latency target <10 ms (primary)
  - CPU latency target <50 ms (secondary)
  - Extended step configs: 10, 25, 50, 100, 500 steps
  - Quality-latency trade-off curve plots (FR-702 / Gap 4)
"""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FR-501 / FR-502 / FR-503  Core latency + memory + CPU benchmarker
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_latency(
    fn: Callable,
    *args,
    n_iterations: int = 100,
    warmup: int = 10,
    device: str = "cpu",
    **kwargs,
) -> dict:
    """
    Measure wall-clock latency of a callable over N iterations.

    Parameters
    ----------
    fn           : callable to benchmark (e.g. model forward / filter fn)
    *args        : positional args passed to fn
    n_iterations : number of measured iterations
    warmup       : warm-up iterations (not included in stats)
    device       : 'cuda' or 'cpu'

    Returns
    -------
    dict with keys: mean_ms, std_ms, min_ms, max_ms, p95_ms, throughput_sps
    """
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            fn(*args, **kwargs)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    times_ms: List[float] = []
    for _ in range(n_iterations):
        if device == "cuda" and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                fn(*args, **kwargs)
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                fn(*args, **kwargs)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    mean_ms = float(np.mean(arr))
    result = {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "throughput_sps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
    }
    return result


def benchmark_memory(
    fn: Callable,
    *args,
    device: str = "cpu",
    **kwargs,
) -> dict:
    """
    Measure peak RAM / VRAM usage during a single forward call.

    Returns
    -------
    dict with keys: peak_ram_mb (always), peak_vram_mb (CUDA only)
    """
    import tracemalloc

    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tracemalloc.start()
    with torch.no_grad():
        fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {"peak_ram_mb": peak / (1024**2)}
    if device == "cuda" and torch.cuda.is_available():
        result["peak_vram_mb"] = torch.cuda.max_memory_allocated() / (1024**2)

    return result


def benchmark_cpu_utilization(
    fn: Callable,
    *args,
    duration_s: float = 5.0,
    **kwargs,
) -> dict:
    """
    Measure CPU utilization % while running fn in a loop for duration_s seconds.

    FR-503 – psutil monitoring.

    Returns
    -------
    dict with keys: mean_cpu_pct, max_cpu_pct, n_threads
    """
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:
        logger.warning("psutil not installed; skipping CPU util measurement.")
        return {"mean_cpu_pct": None, "max_cpu_pct": None, "n_threads": None}

    cpu_readings: List[float] = []
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        with torch.no_grad():
            fn(*args, **kwargs)
        cpu_readings.append(proc.cpu_percent(interval=None))

    return {
        "mean_cpu_pct": float(np.mean(cpu_readings)) if cpu_readings else None,
        "max_cpu_pct": float(np.max(cpu_readings)) if cpu_readings else None,
        "n_threads": proc.num_threads(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FR-505  Comprehensive benchmark suite
# ─────────────────────────────────────────────────────────────────────────────

def run_full_benchmark_suite(
    models: Dict[str, nn.Module],
    baselines: Dict[str, Callable],
    sample_input_torch: torch.Tensor,   # shape (1, 1, 750)
    sample_input_numpy: np.ndarray,     # shape (1, 1, 750)
    *,
    n_iterations: int = 100,
    warmup: int = 10,
    device: str = "cuda",
) -> List[dict]:
    """
    Benchmark all models and baselines; return a list of result dicts.

    Parameters
    ----------
    models     : dict name→nn.Module (e.g. 'diffusion_10', 'cnn_student')
    baselines  : dict name→Callable (e.g. 'butterworth', 'wavelet')
    sample_input_torch : GPU/CPU tensor for PyTorch models
    sample_input_numpy : NumPy array for classical baselines
    n_iterations : iterations per benchmark
    device       : 'cuda' or 'cpu'

    Returns
    -------
    list of dicts, each row is one model configuration
    """
    results = []

    # PyTorch models
    for name, model in models.items():
        model = model.to(device)
        model.eval()
        x = sample_input_torch.to(device)

        logger.info("Benchmarking model: %s", name)
        lat = benchmark_latency(model, x, n_iterations=n_iterations,
                                warmup=warmup, device=device)
        mem = benchmark_memory(model, x, device=device)

        row = {"model": name, "type": "neural", **lat, **mem}
        results.append(row)
        logger.info(
            "  %-30s  mean=%.2f ms  p95=%.2f ms  RAM=%.1f MB",
            name, lat["mean_ms"], lat["p95_ms"], mem["peak_ram_mb"],
        )

    # Classical baselines (NumPy)
    for name, fn in baselines.items():
        logger.info("Benchmarking baseline: %s", name)
        lat = benchmark_latency(fn, sample_input_numpy,
                                n_iterations=n_iterations, warmup=warmup)
        mem = benchmark_memory(fn, sample_input_numpy)

        row = {"model": name, "type": "classical", **lat, **mem}
        results.append(row)
        logger.info(
            "  %-30s  mean=%.2f ms  p95=%.2f ms  RAM=%.1f MB",
            name, lat["mean_ms"], lat["p95_ms"], mem["peak_ram_mb"],
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FR-701  Save results CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(results: List[dict], path: str) -> None:
    """Save benchmark results list to a CSV file."""
    import csv

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Results saved to %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# FR-702 / Gap 4  Quality-latency trade-off curve  – NEW
# ─────────────────────────────────────────────────────────────────────────────

def plot_quality_latency_curve(
    results: List[dict],
    output_path: str = "results/plots/quality_latency_curve.png",
) -> None:
    """
    Generate quality (SNR improvement dB) vs latency (ms) scatter / line plot.

    Each point = one model configuration.
    Annotates each point with its model name.

    FR-702 / SRS Gap 4: documented quality-latency trade-off curves.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    latencies = [r.get("mean_ms", 0) for r in results]
    snrs = [r.get("snr_improvement_db", 0) for r in results]
    names = [r.get("model", "") for r in results]
    types_ = [r.get("type", "neural") for r in results]

    colors = ["#4CAF50" if t == "classical" else "#2196F3" for t in types_]

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(latencies, snrs, c=colors, s=120, zorder=5, edgecolors="white", linewidths=0.8)

    for x, y, name in zip(latencies, snrs, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color="#333333")

    # Sort by latency and draw line for neural models only
    neural = [(lat, snr, n) for lat, snr, n, t in zip(latencies, snrs, names, types_) if t == "neural"]
    if neural:
        neural.sort()
        nx, ny, _ = zip(*neural)
        ax.plot(nx, ny, "--", color="#2196F3", alpha=0.5, linewidth=1.2, label="Neural models")

    ax.set_xlabel("Inference Latency (ms)", fontsize=12)
    ax.set_ylabel("SNR Improvement (dB)", fontsize=12)
    ax.set_title("Quality–Latency Trade-off Curve\nEEG Denoising Models", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Classical baseline"),
        Patch(facecolor="#2196F3", label="Neural (diffusion/distilled)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Quality-latency curve saved to %s", output_path)


def plot_latency_comparison(
    results: List[dict],
    output_path: str = "results/plots/latency_comparison.png",
) -> None:
    """Horizontal bar chart comparing inference latencies across all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    names = [r.get("model", "") for r in results]
    means = [r.get("mean_ms", 0) for r in results]
    p95s = [r.get("p95_ms", 0) for r in results]
    types_ = [r.get("type", "neural") for r in results]
    colors = ["#4CAF50" if t == "classical" else "#2196F3" for t in types_]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5 + 1)))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, means, color=colors, edgecolor="white", linewidth=0.6)
    ax.errorbar(means, y_pos, xerr=[np.array(means) - np.array(p95s)
                                    if False else [0]*len(means),
                                    np.array(p95s) - np.array(means)],
                fmt="none", color="gray", capsize=4, linewidth=1.2)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Inference Latency (ms)", fontsize=11)
    ax.set_title("Model Latency Comparison", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, axis="x", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, means):
        ax.text(val * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Latency comparison plot saved to %s", output_path)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str = "model",
    output_path: str = "results/plots/training_curve.png",
) -> None:
    """Plot training and validation loss curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="Train loss", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val loss", color="#FF5722", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (MSE)", fontsize=11)
    ax.set_title(f"Training Curve – {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training curve saved to %s", output_path)
