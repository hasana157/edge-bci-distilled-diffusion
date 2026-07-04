from __future__ import annotations

import csv
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
REPORT_ASSETS = ROOT / "report_assets"
ZIP_PATH = ROOT / "results_artifacts.zip"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_latency_vs_snr(rows: list[dict[str, str]]) -> Path:
    path = PLOTS / "latency_vs_snr_plot.png"
    usable = [
        row
        for row in rows
        if row.get("mean_ms") and row.get("snr_improvement_db")
    ]
    labels = [row["model"] for row in usable]
    latency = [float(row["mean_ms"]) for row in usable]
    snr = [float(row["snr_improvement_db"]) for row in usable]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.scatter(latency, snr, s=80, color="#246bfe")
    for label, x_value, y_value in zip(labels, latency, snr):
        ax.annotate(label, (x_value, y_value), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_title("Latency vs. SNR Improvement")
    ax.set_xlabel("Mean latency (ms)")
    ax.set_ylabel("SNR improvement (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_ablation(rows: list[dict[str, str]]) -> Path:
    path = PLOTS / "distillation_ablation.png"
    selected = [
        row
        for row in rows
        if row["model"] in {"diffusion_100steps", "cnn_student", "autoencoder_student", "consistency_student"}
    ]
    labels = [row["model"].replace("_", "\n") for row in selected]
    latency = [float(row["mean_ms"]) for row in selected]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    bars = ax.bar(labels, latency, color=["#7c6dff", "#13a579", "#ffb02e", "#ec5d57"])
    ax.set_title("Distillation Ablation: Teacher vs. Student Latency")
    ax.set_ylabel("Mean latency (ms)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_closed_loop(rows: list[dict[str, str]]) -> Path:
    path = PLOTS / "closed_loop_accuracy.png"
    labels = [row["method"].replace("_", "\n") for row in rows]
    accuracy = [float(row["accuracy"]) * 100 for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    bars = ax.bar(labels, accuracy, color=["#4c7bd9", "#2d9c69", "#f07f2f"])
    ax.set_title("Closed-loop Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(40, max(accuracy) + 8))
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def make_hardware_table(rows: list[dict[str, str]]) -> Path:
    path = PLOTS / "hardware_footprint_table.csv"
    fields = ["model", "p95_latency_ms", "throughput_sps", "peak_ram_mb", "peak_vram_mb"]
    table_rows = [
        {
            "model": row["model"],
            "p95_latency_ms": round(float(row["p95_ms"]), 4),
            "throughput_sps": round(float(row["throughput_sps"]), 4),
            "peak_ram_mb": round(float(row["peak_ram_mb"]), 4),
            "peak_vram_mb": round(float(row["peak_vram_mb"]), 4) if row.get("peak_vram_mb") else "",
        }
        for row in rows
    ]
    write_csv(path, table_rows, fields)
    return path


def copy_required_files() -> list[Path]:
    dashboard_src = REPORT_ASSETS / "dashboard_overview.png"
    if not dashboard_src.exists():
        dashboard_src = ROOT / "dashboarsd ss" / "1.png"

    copied = []
    copies = {
        RESULTS / "benchmark_results.csv": PLOTS / "benchmark_summary.csv",
        dashboard_src: PLOTS / "dashboard_screenshot.png",
    }
    for src, dst in copies.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing required source artifact: {src}")
        shutil.copyfile(src, dst)
        copied.append(dst)
    return copied


def mirror_into_results(paths: list[Path]) -> None:
    for path in paths:
        shutil.copyfile(path, RESULTS / path.name)


def make_zip(paths: list[Path]) -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    benchmark_rows = read_csv(RESULTS / "benchmark_results.csv")
    closed_loop_rows = read_csv(RESULTS / "closed_loop_impact.csv")

    artifact_paths = [
        save_latency_vs_snr(benchmark_rows),
        save_ablation(benchmark_rows),
        save_closed_loop(closed_loop_rows),
        make_hardware_table(benchmark_rows),
        *copy_required_files(),
    ]
    mirror_into_results(artifact_paths)
    make_zip(artifact_paths)

    print(f"Created {ZIP_PATH}")
    for path in artifact_paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
