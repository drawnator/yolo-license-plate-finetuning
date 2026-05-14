# YOLO License Plate Fine-tuning – GPU Dockerfile
# ================================================
# Base image: CUDA 12.1 + cuDNN 8 + Ubuntu 22.04
# Adjust the CUDA version tag to match your host driver.

FROM nvidia/cuda:13.0.3-cudnn-runtime-ubi8

LABEL maintainer="drawnator"
LABEL description="YOLO license plate fine-tuning environment (GPU)"

# ── System packages ────────────────────────────────────────────────────────────
# ENV DEBIAN_FRONTEND=noninteractive
RUN yum update -y && \
    yum install -y \
    python3.11 \
    python3.11-pip \
    && rm -rf /var/cache/yum/*

# RUN ln -s /usr/bin/python3.11 /usr/bin/python3
RUN ln -s /usr/bin/python3.11 /usr/bin/python
# ── Python dependencies ────────────────────────────────────────────────────────

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────────────
COPY . .

# ── Default command ────────────────────────────────────────────────────────────
CMD ["python", "-c", "from ultralytics import YOLO; print('YOLO GPU environment ready.')"]
CMD ["nvidia-smi"]
CMD ["python", "prep_dataset.py"]


