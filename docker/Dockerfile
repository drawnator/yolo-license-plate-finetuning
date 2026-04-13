# YOLO License Plate Fine-tuning – GPU Dockerfile
# ================================================
# Base image: CUDA 12.1 + cuDNN 8 + Ubuntu 22.04
# Adjust the CUDA version tag to match your host driver.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="drawnator"
LABEL description="YOLO license plate fine-tuning environment (GPU)"

# ── System packages ────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        git \
        wget \
        curl \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ── Python dependencies ────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────────────
COPY . .

# ── Default command ────────────────────────────────────────────────────────────
CMD ["python", "-c", "from ultralytics import YOLO; print('YOLO GPU environment ready.')"]
