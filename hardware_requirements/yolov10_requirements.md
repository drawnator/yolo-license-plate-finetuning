# YOLOv10 Hardware Requirements

## Overview

YOLOv10 introduces NMS-free end-to-end detection, which reduces post-processing
latency. It requires slightly more VRAM than YOLOv8 for equivalent model tiers.

---

## Hardware Tiers

### Minimum (YOLOv10n / YOLOv10s — small datasets, batch 8–16)

| Component | Specification                          |
|-----------|----------------------------------------|
| GPU       | NVIDIA RTX 2070 8 GB / RTX 3060 8 GB  |
| VRAM      | 8 GB                                   |
| CPU       | Intel i7 / AMD Ryzen 7 (8+ cores)      |
| RAM       | 16 GB DDR4                             |
| Storage   | 25 GB SSD                              |
| OS        | Ubuntu 20.04 / Windows 10 (WSL2)       |

### Recommended (YOLOv10m / YOLOv10l — medium datasets, batch 16–32)

| Component | Specification                           |
|-----------|-----------------------------------------|
| GPU       | NVIDIA RTX 3080 10 GB / RTX 3090 24 GB |
| VRAM      | 10–24 GB                               |
| CPU       | Intel i7 / AMD Ryzen 7 (12+ cores)     |
| RAM       | 32 GB DDR4                             |
| Storage   | 40 GB NVMe SSD                         |
| OS        | Ubuntu 20.04+ / Windows 11 (WSL2)      |

### Optimal (YOLOv10x — large datasets, batch 32+)

| Component | Specification                          |
|-----------|----------------------------------------|
| GPU       | NVIDIA RTX 4090 24 GB / A100 80 GB    |
| VRAM      | 24–80 GB                              |
| CPU       | Intel i9 / AMD Ryzen 9 (16+ cores)    |
| RAM       | 64 GB DDR5                            |
| Storage   | 60 GB NVMe SSD                        |
| OS        | Ubuntu 22.04                          |

---

## Model Size Comparison

| Model    | Params | mAP50-95 (COCO) | Approx. GPU Req. |
|----------|--------|------------------|------------------|
| YOLOv10n | 2.3M   | 38.5             | 6 GB VRAM        |
| YOLOv10s | 7.2M   | 46.3             | 8 GB VRAM        |
| YOLOv10m | 15.4M  | 51.1             | 10 GB VRAM       |
| YOLOv10b | 19.1M  | 52.5             | 12 GB VRAM       |
| YOLOv10l | 24.4M  | 53.2             | 16 GB VRAM       |
| YOLOv10x | 29.5M  | 54.4             | 20+ GB VRAM      |

---

## Installation

```bash
pip install ultralytics>=8.2.0
```

## Training Command

```bash
python training/train_yolov10.py \
  --data      data.yaml \
  --model     yolov10s.pt \
  --img-size  640 \
  --batch-size 16 \
  --epochs    100 \
  --device    0
```
