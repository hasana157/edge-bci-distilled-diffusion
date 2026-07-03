# FAQ

## Can I run this without downloading the EEG dataset?

Yes. The data pipeline generates deterministic synthetic EEG when BCI Competition IV 2a `.mat` files are absent. This is only for smoke tests and development. Do not report synthetic fallback numbers as research results.

## Where do pretrained checkpoints live?

Large `.pt`, `.pth`, and raw dataset files are excluded from Git. Use `models/README.md` for the expected local layout, and publish trained checkpoints through GitHub Releases, Google Drive, or Hugging Face Hub.

## Why use a diffusion teacher if students do the edge inference?

The teacher provides high-quality denoising targets during training. The deployed model is usually a distilled student, which uses one forward pass and is much faster than iterative reverse diffusion.

## What latency does the benchmark measure?

The benchmark measures warm-cache wall-clock inference time after warmup. It excludes dataset download and training time. See `BENCHMARKING_PROTOCOL.md` for the exact input shape, iteration count, and reported statistics.

## Are the committed baseline results final claims?

No. `results/baseline_metrics.csv` is a small sanity-check artifact. Final claims should be regenerated with trained checkpoints, saved configs, the full dataset, and the benchmark protocol.

## Which model should I deploy first?

Start with `cnn_student.onnx` for the best latency-quality balance. Use `autoencoder_student.onnx` when memory and CPU budget are tighter than signal quality requirements.
