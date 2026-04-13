# Hardware Requirements

This document summarises the recommended hardware for fine-tuning each YOLO
version included in this repository on a license plate dataset.

---

## YOLOv5

| Tier        | GPU                           | VRAM   | RAM    | Storage |
|-------------|-------------------------------|--------|--------|---------|
| **Minimum** | GTX 1060 / RTX 2060           | 6 GB   | 16 GB  | 20 GB   |
| **Recommended** | RTX 3070 / RTX 3080       | 8–10 GB | 16 GB | 30 GB  |
| **Optimal** | RTX 3090 / A100               | 24 GB  | 32 GB  | 50 GB   |

See [yolov5_requirements.md](yolov5_requirements.md) for full details.

---

## YOLOv8

| Tier        | GPU                           | VRAM   | RAM    | Storage |
|-------------|-------------------------------|--------|--------|---------|
| **Minimum** | GTX 1060 / RTX 2060           | 6 GB   | 16 GB  | 20 GB   |
| **Recommended** | RTX 3070 / RTX 3080       | 8–16 GB | 16 GB | 30 GB  |
| **Optimal** | RTX 3090 / A100               | 24 GB  | 32 GB  | 50 GB   |

See [yolov8_requirements.md](yolov8_requirements.md) for full details.

---

## YOLOv10

| Tier        | GPU                           | VRAM   | RAM    | Storage |
|-------------|-------------------------------|--------|--------|---------|
| **Minimum** | RTX 2070 / RTX 3060           | 8 GB   | 16 GB  | 25 GB   |
| **Recommended** | RTX 3080 / RTX 3090       | 10–24 GB | 32 GB | 40 GB |
| **Optimal** | RTX 4090 / A100               | 24–80 GB | 64 GB | 60 GB |

See [yolov10_requirements.md](yolov10_requirements.md) for full details.

---

## YOLOv11

| Tier        | GPU                           | VRAM   | RAM    | Storage |
|-------------|-------------------------------|--------|--------|---------|
| **Minimum** | RTX 3060 / RTX 3070           | 8 GB   | 32 GB  | 25 GB   |
| **Recommended** | RTX 3090 / RTX 4080       | 16–24 GB | 32 GB | 40 GB |
| **Optimal** | RTX 4090 / A100               | 24–80 GB | 64 GB | 60 GB |

See [yolov11_requirements.md](yolov11_requirements.md) for full details.

---

## YOLOv12

| Tier        | GPU                           | VRAM   | RAM    | Storage |
|-------------|-------------------------------|--------|--------|---------|
| **Minimum** | RTX 3080 / RTX 4070           | 10 GB  | 32 GB  | 30 GB   |
| **Recommended** | RTX 4090 / A6000          | 24 GB  | 64 GB  | 50 GB   |
| **Optimal** | A100 / H100                   | 80 GB  | 128 GB | 100 GB  |

See [yolov12_requirements.md](yolov12_requirements.md) for full details.

---

## Cloud GPU Recommendations

| Provider     | Instance Type   | GPU          | Approx. Cost/hr |
|--------------|-----------------|--------------|-----------------|
| AWS          | `p3.2xlarge`    | Tesla V100   | ~$3.06          |
| AWS          | `p4d.24xlarge`  | 8× A100      | ~$32.77         |
| Google Cloud | `n1-standard-8` + T4 | Tesla T4 | ~$0.95        |
| Google Cloud | `a2-highgpu-1g` | A100         | ~$3.67          |
| Azure        | `NC6s_v3`       | Tesla V100   | ~$3.06          |

> Prices are approximate and subject to change. Always check the provider's
> pricing page for up-to-date information.
