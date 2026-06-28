"""
run_all_experiments.py – Master experiment runner.

FR-704  python run_all_experiments.py  → produces ALL results.

Pipeline:
  1. Data preparation (with synthetic fallback if no .mat files)
  2. Baseline evaluation (Butterworth, Wavelet, Wiener)
  3. Diffusion teacher training
  4. Distillation (CNN + Autoencoder + Consistency)
  5. ONNX export + CPU benchmark
  6. Latency benchmarking (all models, all step configs)
  7. Classifier training
  8. Closed-loop simulation + denoising impact
  9. Plot generation (quality-latency curve, confusion matrices, etc.)
  10. Results CSV export

Usage
-----
    # Colab / GPU:
    python run_all_experiments.py

    # CPU-only (slower):
    python run_all_experiments.py --device cpu --epochs 100

    # Skip training (benchmark only):
    python run_all_experiments.py --skip-training
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch

# ── add src/ to path so imports work from project root ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config
from data_pipeline import build_dataloaders, build_classifier_dataloaders
from baselines import run_all_baselines
from diffusion import GaussianDiffusion, UNet1D, train_diffusion, load_checkpoint
from distillation import (
    CNNStudent, AutoencoderStudent, ConsistencyStudent,
    train_student, hyperparameter_sweep, export_to_onnx, benchmark_onnx,
)
from benchmarking import (
    run_full_benchmark_suite, save_results_csv,
    plot_quality_latency_curve, plot_latency_comparison, plot_training_curves,
)
from classifier import (
    MotorImageryClassifier, train_classifier, evaluate_classifier,
    ClosedLoopSimulator, measure_denoising_impact,
    plot_confusion_matrix, plot_denoising_impact,
)
from metrics import compute_all_signal_metrics, compute_all_classification_metrics, print_metrics_table


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/experiment.log", mode="w"),
    ],
)
logger = logging.getLogger("run_all")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EBC: EEG BCI Distillation Experiments")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--epochs-diffusion", type=int, default=None,
                   help="Override diffusion training epochs")
    p.add_argument("--epochs-distill", type=int, default=None,
                   help="Override distillation epochs")
    p.add_argument("--epochs-classifier", type=int, default=None)
    p.add_argument("--max-epochs-today", type=int, default=None,
                   help="Maximum new epochs to train in this session")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--skip-training", action="store_true",
                   help="Skip training; load existing checkpoints for benchmarking")
    p.add_argument("--skip-sweep", action="store_true",
                   help="Skip hyperparameter sweep (FR-405)")
    p.add_argument("--data-dir", default="data/raw",
                   help="Path to BCI Competition IV 2a .mat files")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    cfg = Config()
    cfg.device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if cfg.device == "cpu":
        logger.warning("CUDA not available – falling back to CPU. Training will be slow.")
        # Reduce epochs for CPU to be practical
        cfg.diffusion.epochs = args.epochs_diffusion or 100
        cfg.diffusion.batch_size = args.batch_size or 16
        cfg.distill.epochs = args.epochs_distill or 200
        cfg.closed_loop.classifier_epochs = args.epochs_classifier or 50
    else:
        cfg.diffusion.epochs = args.epochs_diffusion or 75
        cfg.diffusion.batch_size = args.batch_size or cfg.diffusion.batch_size
        cfg.distill.epochs = args.epochs_distill or 75
        cfg.closed_loop.classifier_epochs = args.epochs_classifier or cfg.closed_loop.classifier_epochs

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs("results/plots", exist_ok=True)
    # Changed default checkpoint dirs to Google Drive
    diff_ckpt_dir = "/content/drive/MyDrive/ebc_checkpoints/diffusion_teacher"
    distill_ckpt_dir = "/content/drive/MyDrive/ebc_checkpoints/distilled"
    os.makedirs(diff_ckpt_dir, exist_ok=True)
    os.makedirs(distill_ckpt_dir, exist_ok=True)
    os.makedirs("models/onnx", exist_ok=True)
    os.makedirs("models/classifier", exist_ok=True)

    logger.info("=" * 60)
    logger.info("EBC: EEG BCI Diffusion Distillation — GPU-Accelerated Version")
    logger.info("Device: %s | Diffusion epochs: %d | Distill epochs: %d",
                cfg.device, cfg.diffusion.epochs, cfg.distill.epochs)
    logger.info("=" * 60)

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    logger.info("Step 1/9: Building dataloaders …")
    loaders = build_dataloaders(
        dataset_dir=args.data_dir,
        signal_length=cfg.data.signal_length,
        n_channels=cfg.data.n_channels,
        snr_db=15.0,
        batch_size=cfg.diffusion.batch_size,
        random_seed=args.seed,
        synthetic_fallback=True,
    )
    clf_loaders = build_classifier_dataloaders(
        dataset_dir=args.data_dir,
        signal_length=cfg.data.signal_length,
        n_channels=cfg.data.n_channels,
        batch_size=cfg.diffusion.batch_size,
        random_seed=args.seed,
        synthetic_fallback=True,
    )

    # Grab a small numpy batch for baseline + benchmarking
    x_noisy_batch, x_clean_batch, _ = next(iter(loaders["test"]))
    clean_np = x_clean_batch.numpy()
    noisy_np = x_noisy_batch.numpy()

    # ── Step 2: Baselines ────────────────────────────────────────────────────
    logger.info("Step 2/9: Running classical baselines …")
    # Reshape to (trials, channels, samples) for baselines
    clean_3d = clean_np.reshape(-1, 1, cfg.data.signal_length)
    noisy_3d = noisy_np.reshape(-1, 1, cfg.data.signal_length)
    baseline_results = run_all_baselines(clean_3d, noisy_3d, fs=250.0)

    baseline_metrics = {}
    for name, r in baseline_results.items():
        baseline_metrics[name] = {
            "snr_improvement_db": r["snr_improvement_db"],
            "mse": r["mse"],
            "mean_ms": None,   # filled by benchmark suite
        }

    # ── Step 3: Diffusion teacher ────────────────────────────────────────────
    logger.info("Step 3/9: Diffusion teacher …")
    diffusion_cfg = cfg.diffusion
    teacher = UNet1D(
        model_channels=diffusion_cfg.model_channels,
        channel_mult=diffusion_cfg.channel_mult,
        num_res_blocks=diffusion_cfg.num_res_blocks,
        dropout=diffusion_cfg.dropout,
        time_emb_dim=diffusion_cfg.time_emb_dim,
    )
    diffusion = GaussianDiffusion(
        n_steps=diffusion_cfg.n_steps,
        beta_start=diffusion_cfg.beta_start,
        beta_end=diffusion_cfg.beta_end,
        schedule=diffusion_cfg.schedule,
    )
    logger.info("Teacher params: %s", f"{teacher.num_parameters:,}")

    teacher_ckpt = os.path.join(diff_ckpt_dir, "best_model.pt")
    if args.skip_training and os.path.exists(teacher_ckpt):
        load_checkpoint(teacher, teacher_ckpt, device=cfg.device)
        logger.info("Loaded existing teacher checkpoint.")
        diff_history = {"train_losses": [], "val_losses": []}
    else:
        diff_history = train_diffusion(
            teacher, diffusion,
            loaders["train"], loaders["val"],
            epochs=cfg.diffusion.epochs,
            max_epochs_today=args.max_epochs_today,
            lr=diffusion_cfg.lr,
            lr_step=diffusion_cfg.lr_step,
            lr_gamma=diffusion_cfg.lr_gamma,
            grad_clip=diffusion_cfg.grad_clip,
            checkpoint_dir=diff_ckpt_dir,
            save_every=diffusion_cfg.save_every,
            early_stop_patience=diffusion_cfg.early_stop_patience,
            device=cfg.device,
        )
        if diff_history["train_losses"]:
            plot_training_curves(
                diff_history["train_losses"], diff_history["val_losses"],
                "Diffusion Teacher", "results/plots/diffusion_training.png",
            )

    # ── Step 4: Students ─────────────────────────────────────────────────────
    logger.info("Step 4/9: Distillation training …")

    # Load best teacher before distillation
    if os.path.exists(teacher_ckpt):
        load_checkpoint(teacher, teacher_ckpt, device=cfg.device)

    students = {
        "cnn_student": CNNStudent(cfg.data.signal_length),
        "autoencoder_student": AutoencoderStudent(cfg.data.signal_length, latent_dim=64),
        "consistency_student": ConsistencyStudent(cfg.data.signal_length),
    }

    distill_histories = {}
    for sname, student in students.items():
        logger.info("  Training student: %s (%s params)", sname, f"{student.num_parameters:,}")
        ckpt_path = os.path.join(distill_ckpt_dir, sname, "best_model.pt")
        if args.skip_training and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=cfg.device)
            student.load_state_dict(ckpt["model_state"])
            logger.info("  Loaded existing student: %s", sname)
            distill_histories[sname] = {"train_losses": [], "val_losses": []}
        else:
            hist = train_student(
                student, teacher, diffusion,
                loaders["train"], loaders["val"],
                epochs=cfg.distill.epochs,
                max_epochs_today=args.max_epochs_today,
                lr=cfg.distill.lr,
                lr_step=cfg.distill.lr_step,
                lr_gamma=cfg.distill.lr_gamma,
                temperature=cfg.distill.temperature,
                alpha=cfg.distill.alpha,
                checkpoint_dir=distill_ckpt_dir,
                model_name=sname,
                save_every=cfg.distill.save_every,
                early_stop_patience=cfg.distill.early_stop_patience,
                device=cfg.device,
            )
            distill_histories[sname] = hist
            if hist["train_losses"]:
                plot_training_curves(
                    hist["train_losses"], hist["val_losses"],
                    f"Student: {sname}",
                    f"results/plots/{sname}_training.png",
                )

    # Optional hyperparameter sweep (FR-405)
    if not args.skip_sweep and not args.skip_training:
        logger.info("  Hyperparameter sweep (FR-405) …")
        sweep_results = hyperparameter_sweep(
            CNNStudent, teacher, diffusion, loaders["train"], loaders["val"],
            lr_values=cfg.distill.lr_sweep,
            alpha_values=cfg.distill.alpha_sweep,
            sweep_epochs=30,
            device=cfg.device,
        )
        logger.info("Sweep top-3: %s", [(r["lr"], r["alpha"], r["best_val_loss"]) for r in sweep_results[:3]])

    # ── Step 5: ONNX Export + CPU Benchmark (FR-350) ─────────────────────────
    logger.info("Step 5/9: ONNX export + CPU inference benchmark (FR-350) …")
    onnx_results = {}
    for sname, student in students.items():
        onnx_path = os.path.join(cfg.distill.onnx_export_dir, f"{sname}.onnx")
        try:
            export_to_onnx(student, onnx_path, signal_length=cfg.data.signal_length)
            onnx_results[sname] = benchmark_onnx(onnx_path, cfg.data.signal_length,
                                                  n_iterations=100, warmup=10)
        except Exception as exc:
            logger.warning("ONNX export/benchmark failed for %s: %s", sname, exc)

    # ── Step 6: Latency benchmarking (FR-501 to FR-505) ─────────────────────
    logger.info("Step 6/9: Comprehensive latency benchmarking …")
    from baselines import butterworth_bandpass, wavelet_denoise, wiener_filter

    # Sample inputs for benchmarking
    sample_torch = x_noisy_batch[:1].to(cfg.device)   # (1, 1, 750)
    sample_np = noisy_np[:1]                           # (1, 1, 750)

    # Build model dict: one entry per step count for diffusion
    def _make_diffusion_fn(steps):
        def fn(x):
            return diffusion.denoise(teacher, x, steps=steps)
        return fn

    # Neural models dict
    bench_models = {}
    for steps in cfg.diffusion.inference_steps:
        bench_models[f"diffusion_{steps}steps"] = type("M", (), {
            "__call__": lambda self, x, s=steps: diffusion.denoise(teacher, x, steps=s),
            "to": lambda self, d: self,
            "eval": lambda self: self,
        })()
    for sname, student in students.items():
        bench_models[sname] = student

    # Classical baselines dict (operate on squeezed numpy)
    bench_baselines = {
        "butterworth": lambda x: butterworth_bandpass(x),
        "wavelet": lambda x: wavelet_denoise(x),
        "wiener": lambda x: wiener_filter(x),
    }

    # Run suite – catch errors gracefully
    bench_results = []
    try:
        bench_results = run_full_benchmark_suite(
            bench_models, bench_baselines,
            sample_torch, sample_np,
            n_iterations=cfg.benchmark.n_iterations,
            warmup=cfg.benchmark.warmup,
            device=cfg.device,
        )
    except Exception as exc:
        logger.warning("Benchmark suite error: %s", exc)

    # Merge SNR data from baselines
    for r in bench_results:
        name = r["model"]
        if name in baseline_results:
            r["snr_improvement_db"] = baseline_results[name]["snr_improvement_db"]

    save_results_csv(bench_results, "results/benchmark_results.csv")

    # Plots
    if bench_results:
        plot_latency_comparison(bench_results, "results/plots/latency_comparison.png")
        plot_quality_latency_curve(bench_results, "results/plots/quality_latency_curve.png")

    # ── Step 7: Classifier training (FR-601) ─────────────────────────────────
    logger.info("Step 7/9: Motor imagery classifier …")
    mi_classifier = MotorImageryClassifier(
        n_channels=cfg.data.n_channels,
        n_samples=cfg.data.signal_length,
        n_classes=cfg.data.n_classes,
    )
    logger.info("Classifier params: %s", f"{mi_classifier.num_parameters:,}")

    clf_ckpt = os.path.join(cfg.closed_loop.classifier_checkpoint_dir, "best_classifier.pt")
    if args.skip_training and os.path.exists(clf_ckpt):
        ckpt = torch.load(clf_ckpt, map_location=cfg.device)
        mi_classifier.load_state_dict(ckpt["model_state"])
        logger.info("Loaded existing classifier checkpoint.")
        clf_history = {}
    else:
        clf_history = train_classifier(
            mi_classifier, clf_loaders["train"], clf_loaders["val"],
            epochs=cfg.closed_loop.classifier_epochs,
            lr=cfg.closed_loop.classifier_lr,
            checkpoint_dir=cfg.closed_loop.classifier_checkpoint_dir,
            device=cfg.device,
        )
        if clf_history.get("train_losses"):
            plot_training_curves(
                clf_history["train_losses"], clf_history["val_losses"],
                "MI Classifier", "results/plots/classifier_training.png",
            )

    # Evaluate + confusion matrix
    test_acc, conf_mat = evaluate_classifier(mi_classifier, clf_loaders["test"], cfg.device)
    logger.info("Classifier test accuracy: %.3f", test_acc)
    plot_confusion_matrix(conf_mat, title=f"Classifier (acc={test_acc:.3f})",
                          output_path="results/plots/confusion_matrix_classifier.png")

    # ── Step 8: Closed-loop simulation (FR-602 / FR-603) ────────────────────
    logger.info("Step 8/9: Closed-loop BCI simulation …")

    # Extract a numpy batch from test loader for simulation
    all_eeg, all_labels = [], []
    for eeg_batch, label_batch in clf_loaders["test"]:
        all_eeg.append(eeg_batch.numpy())
        all_labels.append(label_batch.numpy() + 1)  # back to 1-indexed
    eeg_trials_np = np.concatenate(all_eeg, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    # Best student for denoising in the loop (CNN)
    best_student = students["cnn_student"].to(cfg.device).eval()

    denoisers_to_test = {
        "no_denoise": None,
        "butterworth": None,   # handled inside simulator
        "cnn_student": lambda x: best_student(x),
    }

    impact_results = measure_denoising_impact(
        mi_classifier, eeg_trials_np, labels_np, denoisers_to_test, cfg.device
    )
    plot_denoising_impact(impact_results, "results/plots/denoising_impact.png")
    save_results_csv(
        [{"method": k, **{mk: mv for mk, mv in v.items() if not isinstance(mv, list)}}
         for k, v in impact_results.items()],
        "results/closed_loop_impact.csv",
    )

    # ── Step 9: Summary ──────────────────────────────────────────────────────
    logger.info("Step 9/9: Generating summary …")

    summary_metrics = {
        "butterworth": {
            "snr_improvement_db": baseline_results.get("butterworth", {}).get("snr_improvement_db", 0),
            "mean_latency_ms": next((r["mean_ms"] for r in bench_results if r["model"] == "butterworth"), 0),
        },
        "wavelet": {
            "snr_improvement_db": baseline_results.get("wavelet", {}).get("snr_improvement_db", 0),
            "mean_latency_ms": next((r["mean_ms"] for r in bench_results if r["model"] == "wavelet"), 0),
        },
    }
    for sname in students:
        summary_metrics[sname] = {
            "mean_latency_ms": next((r["mean_ms"] for r in bench_results if r["model"] == sname), 0),
            "val_loss": distill_histories.get(sname, {}).get("best_val_loss", float("nan")),
        }

    print_metrics_table(summary_metrics)

    logger.info("=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE.")
    logger.info("Results saved to: results/")
    logger.info("Plots saved to:   results/plots/")
    logger.info("Models saved to:  models/")
    logger.info("=" * 60)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
