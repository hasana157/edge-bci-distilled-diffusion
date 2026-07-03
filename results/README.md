# Results

This directory stores generated experiment outputs and small reproducibility artifacts.

## Committed Files

| File | Description |
|---|---|
| `baseline_metrics.csv` | Small baseline sanity-check result. |
| `baseline_comparison.png` | Example waveform comparison for the baseline sanity check. |
| `README.md` | Result schema and interpretation notes. |

## Generated Files

| File | Description |
|---|---|
| `benchmark_results.csv` | Latency, memory, throughput, and quality table from the full benchmark suite. |
| `closed_loop_impact.csv` | Classification accuracy and latency for closed-loop denoising variants. |
| `experiment.log` | Full run log when `run_all_experiments.py` is executed. |
| `plots/quality_latency_curve.png` | Quality-latency trade-off curve. |
| `plots/latency_comparison.png` | Latency comparison across models and baselines. |
| `plots/denoising_impact.png` | Closed-loop accuracy with and without denoising. |
| `plots/confusion_matrix_classifier.png` | Motor-imagery classifier confusion matrix. |
| `plots/*_training.png` | Training curves for teacher, students, and classifier. |

## Interpretation

The committed baseline CSV is a smoke artifact, not a final research claim. Full claims should cite:

- Dataset source and split.
- Hardware.
- Dependency versions.
- Random seed.
- Benchmark command.
- Mean and p95 latency.
- SNR improvement and closed-loop accuracy.
