# Contributing

Thanks for improving EBC. This is a research prototype, so contributions are most useful when they preserve reproducibility and make assumptions explicit.

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Pull Request Checklist

- Keep datasets, checkpoints, and generated caches out of Git.
- Add or update tests for public behavior changes.
- Update README/docs when a command, metric, file path, or result claim changes.
- Include the hardware, Python version, and device used for benchmark changes.
- Keep benchmark claims tied to saved CSV/JSON artifacts where possible.

## Issue Reports

Please include:

- Operating system and Python version.
- CPU/GPU or edge device model.
- Exact command that failed.
- Full error output.
- Whether real BCI Competition IV 2a files or synthetic fallback data were used.

## Coding Style

- Prefer small, focused functions with type hints on public APIs.
- Keep imports package-safe: modules should work from `src.*` imports and from scripts that add `src/` to `PYTHONPATH`.
- Add comments only where they explain non-obvious research logic or hardware assumptions.
