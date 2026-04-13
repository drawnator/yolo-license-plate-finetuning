# YOLOv11 Hardware Requirements

## Overview

YOLO11 (referred to as YOLOv11) features a redesigned C3k2 backbone block and
SPPF improvements that yield better accuracy with fewer parameters than YOLOv8
at comparable model sizes. Multi-scale detection is enhanced for small objects
such as license plates.

---

## Hardware Tiers

### Minimum (YOLO11n / YOLO11s — small datasets, batch 8–16)

| Component | Specification                          |
|-----------|----------------------------------------|
| GPU       | NVIDIA RTX 3060 8 GB / RTX 3070 8 GB  |
| VRAM      | 8 GB                                   |
| CPU       | Intel i7 / AMD Ryzen 7 (8+ cores)      |
| RAM       | 32 GB DDR4                             |
| Storage   | 25 GB SSD                              |
| OS        | Ubuntu 20.04 / Windows 11 (WSL2)       |

### Recommended (YOLO11m / YOLO11l — medium datasets, batch 16–32)

| Component | Specification                           |
|-----------|-----------------------------------------|
| GPU       | NVIDIA RTX 3090 24 GB / RTX 4080 16 GB |
| VRAM      | 16–24 GB                               |
| CPU       | Intel i7/i9 / AMD Ryzen 9 (12+ cores)  |
| RAM       | 32–64 GB DDR4                          |
| Storage   | 40 GB NVMe SSD                         |
| OS        | Ubuntu 22.04 / Windows 11 (WSL2)       |

### Optimal (YOLO11x — large datasets, batch 32+)

| Component | Specification                          |
|-----------|----------------------------------------|
| GPU       | NVIDIA RTX 4090 24 GB / A100 80 GB    |
| VRAM      | 24–80 GB                              |
| CPU       | Intel i9 / AMD Threadripper (16+ cores) |
| RAM       | 64 GB DDR5                            |
| Storage   | 60 GB NVMe SSD                        |
| OS        | Ubuntu 22.04                          |

---

## Model Size Comparison

| Model    | Params | mAP50-95 (COCO) | Approx. GPU Req. |
|----------|--------|------------------|------------------|
| YOLO11n  | 2.6M   | 39.5             | 6 GB VRAM        |
| YOLO11s  | 9.4M   | 47.0             | 8 GB VRAM        |
| YOLO11m  | 20.1M  | 51.5             | 10 GB VRAM       |
| YOLO11l  | 25.3M  | 53.4             | 14 GB VRAM       |
| YOLO11x  | 56.9M  | 54.7             | 20+ GB VRAM      |

---

## Installation

```bash
pip install ultralytics>=8.3.0
```

## Training Command

```bash
python training/train_yolov11.py \
  --data      data.yaml \
  --model     yolo11s.pt \
  --img-size  640 \
  --batch-size 16 \
  --epochs    100 \
  --device    0
```
