# T-020 — Reproducible training container.
# Pins training/requirements.txt, exposes `make train MODEL=... DATASET=...`,
# and writes artefacts to a content-hashed path.
#
# Usage:
#   docker build -t neuro-fabric-train -f training/Dockerfile .
#   docker run --gpus all -v $(pwd)/training/artefacts:/app/training/artefacts \
#     neuro-fabric-train make train MODEL=eegconformer DATASET=bciiv2a
#
# For CPU-only runs, omit --gpus all.

FROM python:3.11-slim

# System dependencies for MOABB/MNE/torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin pip and install Python dependencies first (cache layer).
COPY training/requirements.txt /app/training/requirements.txt
RUN pip install --no-cache-dir --upgrade pip==25.1.1 && \
    pip install --no-cache-dir -r /app/training/requirements.txt

# Copy the training package and repo-root scripts.
COPY training/ /app/training/
COPY scripts/ /app/scripts/

# Environment defaults.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MNE_DATA=/app/training/cache/moabb \
    MOABB_DATA=/app/training/cache/moabb

# Create cache and artefacts directories.
RUN mkdir -p /app/training/cache/moabb /app/training/artefacts

# Entrypoint: allow `make train MODEL=... DATASET=...` and direct script access.
ENTRYPOINT ["python", "-m", "training.scripts.run_all"]

# Default: show available models/datasets.
CMD ["--help"]
