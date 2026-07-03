# Changelog

## 0.2.0 - 2026-07-03

- Added publication-readiness documentation: architecture, methodology, FAQ, contributing guide, model cards, license, authors, and reproducibility config.
- Replaced loose dependency ranges with pinned runtime and development requirements.
- Added smoke tests and GitHub Actions CI.
- Made package imports robust for `src.*` use.
- Removed Colab-only checkpoint paths from the default experiment runner.

## 0.1.0 - 2026-06-26

- Initial research prototype for edge BCI distilled diffusion.
- Added data pipeline, classical baselines, diffusion teacher, distilled students, benchmarking, deployment notes, and closed-loop simulation.
