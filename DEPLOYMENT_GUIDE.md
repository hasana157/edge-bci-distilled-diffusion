# Hardware Deployment Guide

This guide covers exporting distilled EEG denoising students to ONNX and running edge-style inference.

## Exported Model Files

After training and export, the expected files are:

```text
models/onnx/
|-- cnn_student.onnx
|-- autoencoder_student.onnx
`-- consistency_student.onnx
```

If these files are absent, train or load the corresponding PyTorch checkpoints and call `export_to_onnx()` from `src/distillation.py`.

## Python Inference

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "models/onnx/cnn_student.onnx",
    providers=["CPUExecutionProvider"],
)

noisy_eeg = np.random.randn(1, 1, 750).astype(np.float32)
input_name = session.get_inputs()[0].name
denoised_eeg = session.run(None, {input_name: noisy_eeg})[0]

print(noisy_eeg.shape)
print(denoised_eeg.shape)
```

Expected shape for both input and output is `(batch, 1, 750)`.

## Hardware Targets

| Hardware | Recommended model | Notes |
|---|---|---|
| Laptop CPU | CNN student | Good default for desktop BCI experiments. |
| Raspberry Pi or ARM CPU | Autoencoder student | Smaller footprint; validate quality. |
| NVIDIA Jetson | CNN or consistency student | Try TensorRT FP16 after ONNX export. |
| Colab or desktop GPU | Any student | Useful for profiling and comparison. |

Treat these as starting points. Publish measured mean and p95 latency for your exact device.

## Dynamic Quantization

```python
from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic(
    "models/onnx/cnn_student.onnx",
    "models/onnx/cnn_student_int8.onnx",
    weight_type=QuantType.QInt8,
)
```

Quantization can reduce model size and latency, but it should be re-evaluated with SNR and closed-loop accuracy metrics.

## TensorRT on Jetson

```bash
trtexec --onnx=models/onnx/cnn_student.onnx \
        --saveEngine=models/tensorrt/cnn_student.engine \
        --fp16
```

Record TensorRT version, JetPack version, device power mode, and batch size with results.

## BCI Pipeline Integration

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "models/onnx/cnn_student.onnx",
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name


def denoise_eeg_channel(raw_750_samples: np.ndarray) -> np.ndarray:
    x = raw_750_samples.reshape(1, 1, 750).astype(np.float32)
    y = session.run(None, {input_name: x})[0]
    return y.reshape(750)
```

For 22-channel trials, apply the single-channel denoiser channel by channel or batch channels as `(22, 1, 750)` if memory allows.

## Validation Checklist

- Confirm input scaling matches training preprocessing.
- Run warmup before timing.
- Report mean and p95 latency.
- Compare SNR improvement before and after quantization.
- Re-run closed-loop classifier accuracy after deployment changes.
