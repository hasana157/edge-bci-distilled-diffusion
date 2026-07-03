# Models

This directory contains local model artifacts generated during training and export. Large weights are intentionally ignored by Git.

## Expected Layout

```text
models/
|-- diffusion_teacher/
|   |-- best_model.pt
|   `-- latest_model.pt
|-- distilled/
|   |-- cnn_student/
|   |   |-- best_model.pt
|   |   `-- latest_model.pt
|   |-- autoencoder_student/
|   |   |-- best_model.pt
|   |   `-- latest_model.pt
|   `-- consistency_student/
|       |-- best_model.pt
|       `-- latest_model.pt
|-- onnx/
|   |-- cnn_student.onnx
|   |-- autoencoder_student.onnx
|   `-- consistency_student.onnx
`-- classifier/
    `-- best_classifier.pt
```

## Loading a Student Checkpoint

```python
import torch

from src.distillation import CNNStudent

student = CNNStudent(signal_length=750)
checkpoint = torch.load(
    "models/distilled/cnn_student/best_model.pt",
    map_location="cpu",
)
student.load_state_dict(checkpoint["model_state"])
student.eval()
```

## Sharing Weights

Do not commit large `.pt` or `.pth` files. Publish checkpoints through GitHub Releases, Google Drive, or Hugging Face Hub, and document the download link plus checksum in this directory.

See `MODEL_CARDS.md` for intended use and limitations.
