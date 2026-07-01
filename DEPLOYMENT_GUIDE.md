# Hardware Deployment Guide
### Edge-BCI Distilled Diffusion — Edge Inference & Optimization Guide

---

## 1. Overview
This guide explains how to deploy the distilled EEG denoising student models on edge hardware
using ONNX Runtime. The exported `.onnx` models support CPU-only inference, making them
suitable for embedded AI processors, Raspberry Pi, and neuromorphic hardware.

---

## 2. Exported Model Files
After training, find the ready-to-deploy models at:
```
models/onnx/
├── cnn_student.onnx          # ~12 MB, best latency-quality balance
├── autoencoder_student.onnx  # ~0.5 MB, ultra-lightweight
└── consistency_student.onnx  # ~5 MB, best single-step quality
```

---

## 3. Running Inference (Python)
```python
import onnxruntime as ort
import numpy as np

# Load the model
sess = ort.InferenceSession(
    "models/onnx/cnn_student.onnx",
    providers=["CPUExecutionProvider"]  # Use "CUDAExecutionProvider" for GPU
)

# Prepare input: shape (batch, 1, 750), dtype float32
noisy_eeg = np.random.randn(1, 1, 750).astype(np.float32)

# Run inference
input_name = sess.get_inputs()[0].name
denoised_eeg = sess.run(None, {input_name: noisy_eeg})[0]

print(f"Input shape:  {noisy_eeg.shape}")
print(f"Output shape: {denoised_eeg.shape}")  # → (1, 1, 750)
```

---

## 4. Expected Latency by Hardware
| Hardware | Model | Expected Latency |
|---|---|---|
| NVIDIA T4 GPU (Colab) | CNN Student | < 2 ms |
| Laptop CPU (Intel i7) | CNN Student | ~8–15 ms ✅ |
| Laptop CPU (Intel i7) | Autoencoder | ~3–5 ms ✅ |
| Raspberry Pi 4 (ARM) | Autoencoder | ~20–35 ms ✅ |
| Raspberry Pi 4 (ARM) | CNN Student | ~40–60 ms ⚠️ |

> **Recommendation for strict edge deployment:** Use `autoencoder_student.onnx` on resource-constrained hardware (Raspberry Pi, microcontrollers). Use `cnn_student.onnx` on laptop/desktop BCI systems.

---

## 5. Installation Requirements
```bash
pip install onnxruntime numpy
# For GPU inference:
pip install onnxruntime-gpu
```

---

## 6. Further Optimization for Neuromorphic / Embedded Processors

### 6.1 INT8 Quantization (reduces model size ~4x, speeds up ~2x)
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "models/onnx/cnn_student.onnx",
    "models/onnx/cnn_student_int8.onnx",
    weight_type=QuantType.QInt8
)
```

### 6.2 OpenVINO (Intel Neural Compute Stick / NCS2)
```bash
mo --input_model models/onnx/cnn_student.onnx --output_dir models/openvino/
```

### 6.3 TensorRT (NVIDIA Jetson / embedded GPU)
```bash
trtexec --onnx=models/onnx/cnn_student.onnx \
        --saveEngine=models/tensorrt/cnn_student.engine \
        --fp16
```

---

## 7. Integration into a BCI Pipeline
```python
# Minimal real-time BCI denoising loop example
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("models/onnx/cnn_student.onnx",
                             providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

def denoise_eeg_trial(raw_eeg_750_samples: np.ndarray) -> np.ndarray:
    """Denoise a single 3-second EEG trial in real time."""
    x = raw_eeg_750_samples.reshape(1, 1, 750).astype(np.float32)
    return sess.run(None, {input_name: x})[0].reshape(750)

# Example usage
raw_trial = np.random.randn(750)   # Replace with real EEG data
clean_trial = denoise_eeg_trial(raw_trial)
```

---

## 8. Open-Source Repository
All training code, model weights, and this deployment guide are publicly available at:
**https://github.com/hasana157/edge-bci-distilled-diffusion**
