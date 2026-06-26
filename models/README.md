# Models

All trained model checkpoints are stored here.

## Directory Layout

```
models/
├── diffusion_teacher/
│   ├── best_model.pt          ← Best teacher by val loss
│   └── ckpt_ep*.pt            ← Periodic checkpoints
│
├── distilled/
│   ├── cnn_student/
│   │   ├── best_model.pt
│   │   └── ckpt_ep*.pt
│   ├── autoencoder_student/
│   │   ├── best_model.pt
│   │   └── ckpt_ep*.pt
│   └── consistency_student/
│       ├── best_model.pt
│       └── ckpt_ep*.pt
│
├── onnx/
│   ├── cnn_student.onnx
│   ├── autoencoder_student.onnx
│   └── consistency_student.onnx
│
└── classifier/
    └── best_classifier.pt
```

## Loading Checkpoints

```python
import torch
from src.distillation import CNNStudent

student = CNNStudent(750)
ckpt = torch.load("models/distilled/cnn_student/best_model.pt", map_location="cpu")
student.load_state_dict(ckpt["model_state"])
student.eval()
```

## Note
Large `.pt` files are not tracked by git (see `.gitignore`).
Upload to Google Drive or Hugging Face Hub for sharing.
