# Model Cards

These cards describe the intended use and limitations of the model families in this repository. Trained weights are not committed to Git.

## Diffusion Teacher

- File location: `models/diffusion_teacher/best_model.pt`
- Code: `src/diffusion.py`
- Architecture: DDPM-style 1-D U-Net.
- Input: `(batch, 1, 750)` noisy EEG channel windows.
- Output: `(batch, 1, 750)` denoised or predicted-noise tensors depending on call path.
- Intended use: high-quality teacher for distillation and fidelity reference.
- Limitations: iterative reverse diffusion is too slow for many edge deployments.

## CNN Student

- File location: `models/distilled/cnn_student/best_model.pt`
- ONNX export: `models/onnx/cnn_student.onnx`
- Code: `src/distillation.py`
- Input/output: `(batch, 1, 750)`.
- Intended use: balanced latency-quality edge denoiser.
- Limitations: quality depends on teacher checkpoint and target dataset distribution.

## Autoencoder Student

- File location: `models/distilled/autoencoder_student/best_model.pt`
- ONNX export: `models/onnx/autoencoder_student.onnx`
- Code: `src/distillation.py`
- Input/output: `(batch, 1, 750)`.
- Intended use: smallest-footprint denoiser for CPU-constrained hardware.
- Limitations: may smooth informative EEG structure more aggressively than CNN or consistency models.

## Consistency Student

- File location: `models/distilled/consistency_student/best_model.pt`
- ONNX export: `models/onnx/consistency_student.onnx`
- Code: `src/distillation.py`
- Input/output: `(batch, 1, 750)`.
- Intended use: single-step neural denoising with stronger capacity than the small autoencoder.
- Limitations: needs careful benchmark validation on target hardware.

## Safety and Scope

These models are research artifacts for EEG denoising experiments. They are not medical devices and should not be used for diagnosis, treatment, or safety-critical closed-loop stimulation without independent validation.
