from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Edge_BCI_Distilled_Diffusion_Report.pdf"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image(path: str, width: float = 6.4 * inch) -> Image:
    img = Image(str(ROOT / path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    return img


def table(data: list[list[str]], col_widths=None, font_size: int = 8) -> Table:
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c2cc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def fmt_ms(value: str) -> str:
    if not value:
        return "N/R"
    return f"{float(value):.2f}"


def fmt_float(value: str) -> str:
    if not value:
        return "N/R"
    return f"{float(value):.2f}"


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5b6770"))
    canvas.drawString(0.75 * inch, 0.45 * inch, "Edge-BCI Distilled Diffusion Technical Report")
    canvas.drawRightString(7.75 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build() -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=21,
            leading=26,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJustify",
            parent=styles["BodyText"],
            alignment=TA_JUSTIFY,
            fontSize=9.4,
            leading=12.2,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallCenter",
            parent=styles["BodyText"],
            alignment=TA_CENTER,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#5b6770"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            alignment=TA_CENTER,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#344054"),
            spaceBefore=4,
            spaceAfter=10,
        )
    )

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        rightMargin=0.68 * inch,
        leftMargin=0.68 * inch,
        topMargin=0.68 * inch,
        bottomMargin=0.68 * inch,
        title="Edge BCI Distilled Diffusion Report",
        author="Hasan and the Edge-BCI Research Team",
    )

    story = []
    h1, h2, body = styles["Heading1"], styles["Heading2"], styles["BodyJustify"]

    story.append(Spacer(1, 0.45 * inch))
    story.append(Paragraph("Academic Header / Logo Placeholder", styles["SmallCenter"]))
    story.append(Paragraph("Edge-Distilled Diffusion for Real-Time Brain-Computer Interfaces", styles["Heading2"]))
    story.append(
        Paragraph(
            "Bridging the Latency Gap: Distilled Diffusion Models for Real-Time Edge-Based Brain-Computer Interfaces",
            styles["TitleCenter"],
        )
    )
    story.append(Paragraph("<b>Authors:</b> Hasan and the Edge-BCI Research Team", styles["Normal"]))
    story.append(Paragraph("<b>Affiliation:</b> [Insert University/Lab Name]", styles["Normal"]))
    story.append(Paragraph("<b>Date:</b> July 2026", styles["Normal"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Abstract", h1))
    story.append(
        Paragraph(
            "Closed-loop brain-computer interfaces (BCIs) require inference latencies below approximately "
            "20 ms to preserve stable neurofeedback and avoid perceptible degradation in user control. "
            "Diffusion models provide strong denoising fidelity, but standard denoising diffusion probabilistic "
            "models require iterative reverse sampling, commonly 100 to 1000 model evaluations, which is "
            "incompatible with embedded BCI hardware. This report documents an edge-distilled diffusion pipeline "
            "that compresses a server-side diffusion teacher into single-pass CNN, autoencoder, and consistency "
            "student denoisers. The system combines BCI Competition IV 2a preprocessing, teacher-student "
            "distillation, ONNX export, latency profiling, and closed-loop decoding validation. Repository "
            "measurements demonstrate sub-millisecond per-channel student kernels and a 13.40 ms closed-loop "
            "CNN denoising path, while full reverse diffusion ranges from 96.43 ms at 10 steps to 4.61 s at "
            "500 steps.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Table of Contents", h1))
    toc_items = [
        "1. Introduction & Problem Statement",
        "2. System Architecture",
        "3. Methodology",
        "4. Experimental Setup",
        "5. Results",
        "6. Dashboard Overview",
        "7. Deployment Guidelines",
        "8. Conclusion & References",
        "Appendix. GitHub Repository",
    ]
    for item in toc_items:
        story.append(Paragraph(item, styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("List of Figures", h1))
    figures = [
        "Figure 1. System Architecture Block Diagram",
        "Figure 2. Latency-Quality Curve",
        "Figure 3. Ablation Study",
        "Figure 4. Closed-loop Accuracy",
        "Figure 5. Dashboard Overview",
    ]
    for item in figures:
        story.append(Paragraph(item, styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("List of Tables", h1))
    for item in ["Table 1. Full Numerical Benchmark Results", "Table 2. Power/Memory and Hardware Profiling"]:
        story.append(Paragraph(item, styles["Normal"]))
    story.append(PageBreak())

    story.append(Paragraph("1. Introduction & Problem Statement", h1))
    story.append(
        Paragraph(
            "Brain-computer interfaces translate neural activity into control signals for external devices. "
            "For users with ALS, stroke-induced impairment, spinal cord injury, or locked-in syndrome, a reliable "
            "BCI can provide communication and assistive-control channels. Motor-imagery EEG is attractive because "
            "it is non-invasive and wearable, but scalp EEG is noisy, non-stationary, and contaminated by ocular, "
            "muscular, and environmental artifacts.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The fundamental problem is a latency-fidelity conflict. Denoising diffusion probabilistic models "
            "can provide high-fidelity signal restoration, but the reverse process requires iterative sampling. "
            "A standard DDPM may require 100 to 1000 U-Net evaluations. This conflicts with edge hardware limits "
            "and with the strict closed-loop BCI requirement that denoising remain below roughly 20 ms so the "
            "complete neurofeedback loop remains responsive.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The project objective is to optimize quality, latency, and power jointly: maximize signal fidelity "
            "measured by SNR/MSE and decoding preservation, while constraining inference latency and hardware "
            "power. The four research objectives are latency characterization, distillation framework design, "
            "edge optimization, and closed-loop validation.",
            body,
        )
    )

    story.append(Paragraph("2. System Architecture", h1))
    story.append(
        Paragraph(
            "The architecture separates offline compression from online inference. The offline path trains a "
            "DDPM teacher and distills it into student models. The online path executes a single ONNX/TensorRT "
            "forward pass, then feeds the denoised signal into a motor-imagery classifier and feedback simulator.",
            body,
        )
    )
    arch_data = [
        ["Layer", "Block", "Description"],
        ["1", "EEG Acquisition", "BCI Competition IV 2a data or simulated stream"],
        ["2", "Preprocessing", "Band-pass filtering, trimming, 22-channel normalization"],
        ["3", "Denoising Engine", "Teacher DDPM offline; CNN/AE/Consistency student online"],
        ["4", "Decoding", "Motor-imagery EEGNet/shallow CNN classifier"],
        ["5", "Output", "Closed-loop command and monitoring dashboard"],
    ]
    story.append(table(arch_data, [0.55 * inch, 1.7 * inch, 4.8 * inch], 8))
    story.append(Paragraph("Figure 1. System architecture block diagram.", styles["Caption"]))

    story.append(Paragraph("3. Methodology", h1))
    story.append(Paragraph("Teacher DDPM", h2))
    story.append(
        Paragraph(
            "The teacher model follows the DDPM formulation. Clean EEG is noised using a linear or cosine "
            "schedule, and a 1-D U-Net predicts the injected noise. The U-Net uses sinusoidal timestep embeddings, "
            "residual 1-D convolutional blocks, group normalization, SiLU activations, downsampling, upsampling, "
            "and skip connections. The processing unit is a single EEG channel window with shape (B, 1, 750).",
            body,
        )
    )
    story.append(Paragraph("Distillation", h2))
    story.append(
        Paragraph(
            "The distillation objective combines teacher mimicry and hard clean-signal supervision. The implemented "
            "loss is a temperature-scaled KL term between flattened teacher and student outputs plus MSE between "
            "the student output and clean target. Progressive distillation reduces teacher steps conceptually from "
            "N to N/2 until one-step inference, while the consistency student maps noisy samples directly to clean "
            "samples in a single residual convolutional pass.",
            body,
        )
    )
    story.append(Paragraph("Edge Optimization", h2))
    story.append(
        Paragraph(
            "Students are exported to ONNX with dynamic batch axes. The exported input is named eeg_noisy and the "
            "output is eeg_denoised. Both use shape (B, 1, 750). ONNX Runtime provides a CPU edge proxy, TensorRT "
            "supports Jetson-class FP16 deployment, and INT8 quantization is available for lower-memory CPU/NPU "
            "targets. Every optimization must be rechecked for latency, SNR, MSE, and closed-loop accuracy.",
            body,
        )
    )

    story.append(Paragraph("4. Experimental Setup", h1))
    story.append(
        Paragraph(
            "The target dataset is BCI Competition IV Dataset 2a: nine subjects, 22 EEG channels, four motor-imagery "
            "classes, 250 Hz sampling, and 750-sample windows. The repository supports subject-level train/validation/"
            "test splits and synthetic fallback data for smoke tests only. Hardware profiling uses PyTorch CUDA for "
            "teacher/student comparison, ONNX Runtime for CPU proxy deployment, and TensorRT/INT8 as target edge paths.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Evaluation metrics include SNR improvement and MSE for fidelity, mean/std/min/max/p95 latency for "
            "runtime, throughput for execution efficiency, and motor-imagery classification accuracy for closed-loop "
            "validity.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("5. Results", h1))
    story.append(Paragraph("Latency-Quality Curve", h2))
    story.append(image("results/plots/quality_latency_curve.png", 6.6 * inch))
    story.append(Paragraph("Figure 2. Latency-quality curve from the benchmark suite.", styles["Caption"]))

    story.append(Paragraph("Ablation Study", h2))
    ablation = [
        ["Model", "Mean latency (ms)", "P95 latency (ms)", "Interpretation"],
        ["CNN Student", "0.40", "0.46", "Fastest balanced student"],
        ["Autoencoder Student", "0.51", "0.60", "Small footprint, compression-oriented"],
        ["Consistency Student", "0.73", "0.91", "Single-step residual denoising"],
    ]
    story.append(table(ablation, [1.6 * inch, 1.2 * inch, 1.2 * inch, 2.9 * inch], 8))
    story.append(Paragraph("Figure 3. Ablation study comparing the distilled student families.", styles["Caption"]))

    story.append(Paragraph("Closed-loop Accuracy", h2))
    story.append(image("results/plots/denoising_impact.png", 6.6 * inch))
    story.append(Paragraph("Figure 4. Closed-loop denoising impact and classifier accuracy.", styles["Caption"]))

    benchmark_rows = read_csv(ROOT / "results" / "benchmark_results.csv")
    bench_table = [["Model", "Type", "Mean ms", "P95 ms", "Throughput", "SNR gain"]]
    for row in benchmark_rows:
        bench_table.append(
            [
                row["model"],
                row["type"],
                fmt_ms(row["mean_ms"]),
                fmt_ms(row["p95_ms"]),
                fmt_float(row["throughput_sps"]),
                fmt_float(row.get("snr_improvement_db", "")),
            ]
        )
    story.append(Paragraph("Table 1. Full numerical benchmark results.", styles["Caption"]))
    story.append(table(bench_table, [1.65 * inch, 0.75 * inch, 0.85 * inch, 0.85 * inch, 1.0 * inch, 0.9 * inch], 6.8))

    power_memory = [
        ["Device/Runtime", "Framework", "Model", "Memory or artifact size", "Power status"],
        ["Server GPU", "PyTorch CUDA", "DDPM Teacher", "VRAM profile required", "High, to measure"],
        ["Laptop/CPU proxy", "ONNX Runtime", "CNN Student", "~13.1 MB data", "Low, to measure"],
        ["Laptop/CPU proxy", "ONNX Runtime", "Autoencoder", "~3.17 MB data", "Low, to measure"],
        ["Laptop/CPU proxy", "ONNX Runtime", "Consistency", "~0.66 MB data", "Low, to measure"],
        ["Jetson/NPU target", "TensorRT/INT8", "Quantized Student", "Backend-specific", "Future work"],
    ]
    story.append(Paragraph("Table 2. Power/memory and hardware profiling table.", styles["Caption"]))
    story.append(table(power_memory, [1.45 * inch, 1.1 * inch, 1.2 * inch, 1.5 * inch, 1.45 * inch], 7.2))

    story.append(PageBreak())
    story.append(Paragraph("6. Dashboard Overview", h1))
    story.append(
        Paragraph(
            "The dashboard provides an operational view of the simulator UI. It includes ONNX model upload, "
            "target edge hardware selection, synthetic EEG generation, edge inference controls, an EEG oscilloscope, "
            "and metric cards for latency, SNR improvement, MSE, and BCI classifier accuracy.",
            body,
        )
    )
    story.append(image("report_assets/dashboard_overview.png", 7.0 * inch))
    story.append(Paragraph("Figure 5. Dashboard screenshot of the simulator UI.", styles["Caption"]))

    story.append(Paragraph("7. Deployment Guidelines", h1))
    story.append(
        Paragraph(
            "Deployment begins with selecting a trained PyTorch student checkpoint, exporting it to ONNX, validating "
            "the input-output shape contract, and profiling on the target runtime. ONNX Runtime is used for CPU "
            "inference, TensorRT for NVIDIA Jetson FP16 acceleration, and dynamic INT8 quantization for memory-limited "
            "CPU/NPU deployments. Acceptance gates include finite outputs, p95 latency below budget, SNR/MSE within "
            "acceptable range, closed-loop accuracy preservation, and thermal/power compliance.",
            body,
        )
    )

    story.append(Paragraph("8. Conclusion & References", h1))
    story.append(
        Paragraph(
            "The report demonstrates that distillation moves EEG denoising from iterative server-side diffusion into "
            "a real-time edge-compatible path. Repository artifacts show diffusion latency increasing from 96.43 ms "
            "for 10 steps to 4.61 s for 500 steps, while single-pass students run below 1 ms per channel. The closed-loop "
            "CNN path remains under the 20 ms p95 target in the released benchmark. Future work should publish final "
            "trained checkpoints, report multi-seed SNR and decoding confidence intervals, measure real device power, "
            "and explore neuromorphic/spiking implementations.",
            body,
        )
    )
    refs = [
        "Wolpaw et al., Brain-computer interfaces for communication and control, Clinical Neurophysiology, 2002.",
        "Pfurtscheller and Neuper, Motor imagery and direct brain-computer communication, Proceedings of the IEEE, 2001.",
        "Brunner et al., BCI Competition 2008 - Graz data set A, 2008.",
        "Ho, Jain, and Abbeel, Denoising diffusion probabilistic models, NeurIPS, 2020.",
        "Song et al., Consistency models, ICML, 2023.",
        "Salimans and Ho, Progressive distillation for fast sampling of diffusion models, ICLR, 2022.",
        "Hinton, Vinyals, and Dean, Distilling the knowledge in a neural network, 2015.",
        "Lawhern et al., EEGNet: a compact CNN for EEG-based BCIs, Journal of Neural Engineering, 2018.",
        "Schirrmeister et al., Deep learning with CNNs for EEG decoding and visualization, Human Brain Mapping, 2017.",
        "Jacob et al., Quantization and training of neural networks for efficient integer-only inference, CVPR, 2018.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles["Normal"]))

    story.append(Paragraph("Appendix: GitHub Repository", h1))
    story.append(
        Paragraph(
            'Public repository: <a href="https://github.com/hasana157/edge-bci-distilled-diffusion">'
            "https://github.com/hasana157/edge-bci-distilled-diffusion</a>",
            body,
        )
    )

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


if __name__ == "__main__":
    build()
    print(OUT)
