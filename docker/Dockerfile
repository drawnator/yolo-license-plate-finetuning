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
    gcc \
    gcc-c++ \
    make \
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
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────────────
COPY . .
RUN if [ ! -d /datasets ]; then mkdir -p /datasets; fi
RUN python3 -c "import torch; print(torch.cuda.is_available())"

# ── Entrypoint: prepare datasets then run the pipeline ─────────────────────────
# Datasets are mounted from the host via docker-compose volume (./datasets:/datasets).
# prep_dataset.py is idempotent — it skips already-downloaded/extracted datasets.
COPY docker/entrypoint-gpu.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# ── Default CMD flags (see python training/train_pipeline.py --help) ──────────
#   --synthetic N          Generate N synthetic plate images (0 = skip)
#   --pretrain             Pretrain a teacher model on real + synthetic data
#   --regenerate-synthetic Force recreation of the synthetic dataset
#   --regenerate-pseudo    Force regeneration of pseudo-labels
#   --skip-pseudo / --skip-final   Skip a stage
#   --teacher-weights PT   Use a specific .pt file as pseudo-labeling teacher
#   --base-model MODEL     Base model (default: yolo26s.pt)
#   --pretrain-epochs N    Epochs for pretraining (default: 50)
#   --final-epochs N       Epochs for final training (default: 100)
#   --batch-size N         Batch size (default: 16)
#   --device DEVICE        CUDA device or 'cpu' (default: 0)
#   --project DIR          Output directory (default: angelicam)
#   --name NAME            Training run name
CMD ["python", "training/train_pipeline.py", "--synthetic", "5000", "--pretrain","--regenerate-synthetic","--regenerate-pseudo"]


