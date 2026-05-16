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
    unzip \
    python3.11 \
    python3.11-pip \
    libxcb \
    libxcb-devel \
    libXrender \
    libXext \
    libSM \
    libX11 \
    mesa-libGL \
    && yum clean all \
    && rm -rf /var/cache/yum/*

RUN if [ ! -L /usr/bin/python3 ]; then ln -s /usr/bin/python3.11 /usr/bin/python3; fi

RUN if [ ! -L /usr/bin/python ]; then ln -s /usr/bin/python3.11 /usr/bin/python; fi
# ── Python dependencies ────────────────────────────────────────────────────────

COPY requirements.txt ./
RUN pip uninstall -y opencv-python && \
    pip install opencv-python-headless
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────────────
COPY . .
RUN if [! -L /datasets ]; then mkdir datasets; fi
RUN python3 -c "import torch; print(torch.cuda.is_available())"
RUN python prep_dataset.py
# ── Default command ────────────────────────────────────────────────────────────
CMD ["python", "-c", "from ultralytics import YOLO; print('YOLO GPU environment ready.')"]


