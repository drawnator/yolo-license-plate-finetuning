# YOLOv8 Hardware Requirements

## Overview

YOLOv8 offers a unified Python API for detection, segmentation, pose estimation
and classification. It trains faster than YOLOv5 on modern GPUs and achieves
higher accuracy at comparable model sizes.

---

## Hardware Tiers

### Minimum (YOLOv8n / YOLOv8s — small datasets, batch 8–16)

| Component | Specification                          |
|-----------|----------------------------------------|
| GPU       | NVIDIA GTX 1060 6 GB / RTX 2060 6 GB  |
| VRAM      | 6 GB                                   |
| CPU       | Intel i5 / AMD Ryzen 5 (6+ cores)      |
| RAM       | 16 GB DDR4                             |
| Storage   | 20 GB SSD                              |
| OS        | Ubuntu 20.04 / Windows 10 (WSL2)       |

### Recommended (YOLOv8m / YOLOv8l — medium datasets, batch 16–32)

| Component | Specification                          |
|-----------|----------------------------------------|
| GPU       | NVIDIA RTX 3070 8 GB / RTX 3080 10 GB |
| VRAM      | 8–16 GB                               |
| CPU       | Intel i7 / AMD Ryzen 7 (8+ cores)     |
| RAM       | 16–32 GB DDR4                         |
| Storage   | 30 GB NVMe SSD                        |
| OS        | Ubuntu 20.04+ / Windows 11 (WSL2)     |

### Optimal (YOLOv8x — large datasets, batch 32+)

| Component | Specification                          |
|-----------|----------------------------------------|
| GPU       | NVIDIA RTX 3090 24 GB / A100 40–80 GB |
| VRAM      | 24–80 GB                              |
| CPU       | Intel i9 / AMD Ryzen 9 (16+ cores)    |
| RAM       | 32–64 GB DDR4/DDR5                    |
| Storage   | 50 GB NVMe SSD                        |
| OS        | Ubuntu 20.04+ / Ubuntu 22.04          |

---

## Model Size Comparison

| Model   | Params | mAP50-95 (COCO) | Approx. GPU Req. |
|---------|--------|------------------|------------------|
| YOLOv8n | 3.2M   | 37.3             | 4 GB VRAM        |
| YOLOv8s | 11.2M  | 44.9             | 6 GB VRAM        |
| YOLOv8m | 25.9M  | 50.2             | 8 GB VRAM        |
| YOLOv8l | 43.7M  | 52.9             | 12 GB VRAM       |
| YOLOv8x | 68.2M  | 53.9             | 16+ GB VRAM      |

---

## Installation

```bash
pip install ultralytics>=8.0.0
```

## Training Command

```bash
python training/train_yolov8.py \
  --data      data.yaml \
  --model     yolov8s.pt \
  --img-size  640 \
  --batch-size 16 \
  --epochs    100 \
  --device    0
```
