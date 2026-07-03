"""Run a lightweight latency benchmark for distilled student models."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking import benchmark_latency, save_results_csv
from src.distillation import AutoencoderStudent, CNNStudent, ConsistencyStudent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark distilled student inference latency.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--signal-length", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/latency_smoke.csv")
    return parser.parse_args()


def build_models(signal_length: int) -> Dict[str, nn.Module]:
    return {
        "cnn_student": CNNStudent(signal_length=signal_length),
        "autoencoder_student": AutoencoderStudent(signal_length=signal_length, latent_dim=64),
        "consistency_student": ConsistencyStudent(signal_length=signal_length),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    x = torch.randn(args.batch_size, 1, args.signal_length, device=device)

    results = []
    for name, model in build_models(args.signal_length).items():
        model = model.to(device).eval()
        stats = benchmark_latency(
            model,
            x,
            n_iterations=args.iterations,
            warmup=args.warmup,
            device=device,
        )
        results.append(
            {
                "model": name,
                "device": device,
                "batch_size": args.batch_size,
                "signal_length": args.signal_length,
                "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                **stats,
            }
        )

    output_path = Path(args.output)
    os.makedirs(output_path.parent or ".", exist_ok=True)
    save_results_csv(results, str(output_path))

    for row in results:
        print(
            f"{row['model']}: mean={row['mean_ms']:.3f} ms, "
            f"p95={row['p95_ms']:.3f} ms, throughput={row['throughput_sps']:.1f} samples/s"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
