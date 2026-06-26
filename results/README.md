# Results

Generated experiment outputs — committed for reproducibility.

## Files

| File | Description |
|------|-------------|
| `benchmark_results.csv` | Full latency / memory / throughput table |
| `closed_loop_impact.csv` | Denoising impact on classification accuracy |
| `experiment.log` | Full training and evaluation log |

## Plots

| File | Description | FR |
|------|-------------|-----|
| `plots/quality_latency_curve.png` | Quality (SNR dB) vs Latency — trade-off curve | FR-702, Gap 4 |
| `plots/latency_comparison.png` | Horizontal bar: inference latency all models | FR-505 |
| `plots/denoising_impact.png` | Accuracy with / without denoising per method | FR-603 |
| `plots/confusion_matrix_classifier.png` | 4-class motor imagery confusion matrix | FR-601 |
| `plots/diffusion_training.png` | Diffusion teacher loss curve | FR-302 |
| `plots/cnn_student_training.png` | CNN student distillation loss curve | FR-401 |
| `plots/autoencoder_student_training.png` | AE student training curve | FR-403 |
| `plots/consistency_student_training.png` | Consistency student training curve | FR-406 |
| `plots/classifier_training.png` | MI classifier training curve | FR-601 |
