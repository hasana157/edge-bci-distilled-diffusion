# Docker image for the Edge-BCI Flask inference API.
# Targets Render, DigitalOcean, AWS, or any Linux VPS.
# Also suitable for Raspberry Pi 4 / Jetson (use linux/arm64 platform).
#
# Build:  docker build -t edge-bci .
# Run:    docker run -p 5000:5000 edge-bci
# Prod:   docker run -p 5000:5000 -e PORT=5000 edge-bci

FROM python:3.10-slim

# System dependencies for numpy / onnxruntime native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached until requirements change)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy project source — models/onnx must be present in the build context
COPY app.py .
COPY src/ ./src/
COPY models/onnx/ ./models/onnx/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 5000

# gunicorn: single worker keeps memory low on small VPS / edge devices
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "60"]
