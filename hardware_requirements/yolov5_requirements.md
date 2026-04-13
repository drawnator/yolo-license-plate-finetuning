# YOLOv5 Hardware Requirements

## Overview

YOLOv5 is the most accessible YOLO version for fine-tuning. Its smaller model
variants (nano, small) can run on consumer-grade GPUs with modest VRAM.

---

## Hardware Tiers

### Minimum (YOLOv5n / YOLOv5s — small datasets, batch 8–16)

| Component | Specification                          |
|-----------|----------------------------------------|
| GPU       | NVIDIA GTX 1060 6 GB / RTX 2060 6 GB  |
| VRAM      | 6 GB                                   |
| CPU       | Intel i5 / AMD Ryzen 5 (6+ cores)      |
| RAM       | 16 GB DDR4                             |
| Storage   | 20 GB SSD                              |
| OS        | Ubuntu 20.04 / Windows 10 (WSL2)       |

### Recommended (YOLOv5m / YOLOv5l — medium datasets, batch 16–32)

| Component | Specification                          |
|-----------|----------------------------------------|
| GPU       | NVIDIA RTX 3070 8 GB / RTX 3080 10 GB |
| VRAM      | 8–10 GB                               |
| CPU       | Intel i7 / AMD Ryzen 7 (8+ cores)     |
| RAM       | 16–32 GB DDR4                         |
| Storage   | 30 GB NVMe SSD                        |
| OS        | Ubuntu 20.04+ / Windows 11 (WSL2)     |

### Optimal (YOLOv5x — large datasets, batch 32+)

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

| Model      | Params | mAP50-95 (COCO) | Approx. GPU Req. |
|------------|--------|------------------|------------------|
| YOLOv5n    | 1.9M   | 28.0             | 4 GB VRAM        |
| YOLOv5s    | 7.2M   | 37.4             | 6 GB VRAM        |
| YOLOv5m    | 21.2M  | 45.4             | 8 GB VRAM        |
| YOLOv5l    | 46.5M  | 49.0             | 12 GB VRAM       |
| YOLOv5x    | 86.7M  | 50.7             | 16+ GB VRAM      |

---

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/ultralytics/yolov5.git
pip install -r yolov5/requirements.txt
```

## Training Command

```bash
python training/train_yolov5.py \
  --data      data.yaml \
  --weights   yolov5s.pt \
  --img-size  640 \
  --batch-size 16 \
  --epochs    100 \
  --device    0
```
