# YOLOv12 Hardware Requirements

## Overview

YOLOv12 integrates an attention-centric "area attention" mechanism that enables
efficient global context capture. It demands more compute than previous YOLO
versions but delivers state-of-the-art accuracy, particularly on complex scenes
and small objects such as license plates at a distance.

---

## Hardware Tiers

### Minimum (YOLOv12n / YOLOv12s — small datasets, batch 8–16)

| Component | Specification                            |
|-----------|------------------------------------------|
| GPU       | NVIDIA RTX 3080 10 GB / RTX 4070 12 GB  |
| VRAM      | 10–12 GB                                |
| CPU       | Intel i7 / AMD Ryzen 7 (8+ cores)       |
| RAM       | 32 GB DDR4                              |
| Storage   | 30 GB NVMe SSD                          |
| OS        | Ubuntu 22.04 / Windows 11 (WSL2)        |

### Recommended (YOLOv12m / YOLOv12l — medium datasets, batch 16–32)

| Component | Specification                           |
|-----------|-----------------------------------------|
| GPU       | NVIDIA RTX 4090 24 GB / A6000 48 GB    |
| VRAM      | 24–48 GB                               |
| CPU       | Intel i9 / AMD Ryzen 9 (16+ cores)     |
| RAM       | 64 GB DDR5                             |
| Storage   | 50 GB NVMe SSD                         |
| OS        | Ubuntu 22.04                           |

### Optimal (YOLOv12x — large datasets, batch 32+)

| Component | Specification                          |
|-----------|----------------------------------------|
| GPU       | NVIDIA A100 80 GB / H100 80 GB        |
| VRAM      | 80 GB                                 |
| CPU       | AMD Threadripper / Xeon (32+ cores)   |
| RAM       | 128–256 GB ECC DDR5                   |
| Storage   | 100 GB NVMe SSD                       |
| OS        | Ubuntu 22.04                          |

---

## Model Size Comparison

| Model    | Params | mAP50-95 (COCO) | Approx. GPU Req. |
|----------|--------|------------------|------------------|
| YOLOv12n | 2.6M   | 40.6             | 8 GB VRAM        |
| YOLOv12s | 9.3M   | 48.0             | 10 GB VRAM       |
| YOLOv12m | 20.2M  | 53.0             | 16 GB VRAM       |
| YOLOv12l | 26.4M  | 55.2             | 24 GB VRAM       |
| YOLOv12x | 59.1M  | 57.0             | 40+ GB VRAM      |

---

## Installation

```bash
pip install ultralytics>=8.3.0
```

## Training Command

```bash
python training/train_yolov12.py \
  --data      data.yaml \
  --model     yolov12s.pt \
  --img-size  640 \
  --batch-size 16 \
  --epochs    100 \
  --device    0
```
