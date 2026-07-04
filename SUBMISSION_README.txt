Edge-BCI Distilled Diffusion - Supplementary README

Main GitHub repository:
https://github.com/hasana157/edge-bci-distilled-diffusion

Reproduce results:
python run_all_experiments.py

Export distilled models to ONNX:
python src/distillation.py --export_onnx

Hardware used:
Tested on NVIDIA Jetson Orin Nano, 4GB RAM, and x86 CPU with ONNX Runtime

Dependencies:
Python 3.9+
PyTorch 2.0+
ONNX Runtime 1.15+

Submission artifacts:
- Technical report: Edge_BCI_Distilled_Diffusion_Report.pdf
- Experimental artifacts: results_artifacts.zip
- Repository results: results/
- Repository plots: plots/
