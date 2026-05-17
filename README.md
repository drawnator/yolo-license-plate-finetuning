# YOLO License Plate Fine-tuning

A complete, production-ready repository for fine-tuning multiple YOLO versions
on a custom license plate detection dataset. Includes separate training scripts
per YOLO version, Docker support, model evaluation utilities, and
inference/prediction examples.

---

## Table of Contents

- [Supported YOLO Versions](#supported-yolo-versions)
- [Repository Structure](#repository-structure)
- [Hardware Requirements](#hardware-requirements)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference & Prediction](#inference--prediction)
- [Docker](#docker)
- [Augmentation Configuration](#augmentation-configuration)

---

## Supported YOLO Versions

| Version  | Script                            | Min VRAM | Notes                              |
|----------|-----------------------------------|----------|------------------------------------|
| YOLOv5   | `training/train_yolov5.py`        | 6 GB     | Stable, well-documented            |
| YOLOv8   | `training/train_yolov8.py`        | 6 GB     | Unified API, recommended default   |
| YOLOv10  | `training/train_yolov10.py`       | 8 GB     | NMS-free end-to-end detection      |
| YOLOv11  | `training/train_yolov11.py`       | 8 GB     | Improved backbone, fewer params    |
| YOLOv12  | `training/train_yolov12.py`       | 10 GB    | Attention-centric architecture     |

---

## Repository Structure

```
yolo-license-plate-finetuning/
├── data.yaml                         # Dataset configuration
├── requirements.txt                  # Python dependencies
│
├── training/
│   ├── train_yolov5.py               # YOLOv5 fine-tuning script
│   ├── train_yolov8.py               # YOLOv8 fine-tuning script
│   ├── train_yolov10.py              # YOLOv10 fine-tuning script
│   ├── train_yolov11.py              # YOLOv11 fine-tuning script
│   └── train_yolov12.py              # YOLOv12 fine-tuning script
│
├── evaluation/
│   ├── evaluate.py                   # Single-model evaluation
│   └── compare_models.py             # Multi-model comparison + chart
│
├── inference/
│   ├── predict_image.py              # Single-image prediction
│   ├── predict_batch.py              # Batch prediction (folder of images)
│   ├── predict_video.py              # Video file prediction
│   └── predict_realtime.py          # Real-time webcam / RTSP stream
│
├── docker/
│   ├── Dockerfile                    # GPU image (CUDA 12.1)
│   ├── Dockerfile.cpu                # CPU-only image
│   └── docker-compose.yml           # Compose services
│
├── configs/
│   └── augmentation.yaml            # Augmentation hyperparameters
│
├── hardware_requirements/
│   ├── README.md                    # Summary table for all versions
│   ├── yolov5_requirements.md
│   ├── yolov8_requirements.md
│   ├── yolov10_requirements.md
│   ├── yolov11_requirements.md
│   └── yolov12_requirements.md
│
└── data/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

---

## Hardware Requirements

See [`hardware_requirements/README.md`](hardware_requirements/README.md) for a
full breakdown. Quick summary:

| Version  | Min VRAM | Recommended VRAM | Min RAM |
|----------|----------|-------------------|---------|
| YOLOv5   | 6 GB     | 8–10 GB           | 16 GB   |
| YOLOv8   | 6 GB     | 8–16 GB           | 16 GB   |
| YOLOv10  | 8 GB     | 10–24 GB          | 32 GB   |
| YOLOv11  | 8 GB     | 16–24 GB          | 32 GB   |
| YOLOv12  | 10 GB    | 24 GB             | 32 GB   |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

Follow the [Dataset Preparation](#dataset-preparation) section below.

### 3. Train a model

```bash
# YOLOv8 (recommended)
python training/train_yolov8.py

# YOLOv5
python training/train_yolov5.py

# YOLOv10
python training/train_yolov10.py

# YOLOv11
python training/train_yolov11.py

# YOLOv12
python training/train_yolov12.py
```

### 4. Evaluate

```bash
python evaluation/evaluate.py \
  --weights runs/train/yolov8_license_plate/weights/best.pt \
  --data    data.yaml
```

### 5. Run inference

```bash
python inference/predict_image.py \
  --weights runs/train/yolov8_license_plate/weights/best.pt \
  --source  path/to/image.jpg
```

---

## Dataset Preparation

Your dataset must follow the YOLO annotation format.

### Directory layout

```
data/
├── images/
│   ├── train/   # Training images (.jpg / .png)
│   ├── val/     # Validation images
│   └── test/    # Test images (optional)
└── labels/
    ├── train/   # Annotation .txt files
    ├── val/
    └── test/
```

### Label format

Each `.txt` file contains one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are **normalised** (0–1) relative to image dimensions.

**Example** (single license plate):
```
0 0.512 0.347 0.215 0.063
```

### Recommended split

| Split      | Percentage |
|------------|-----------|
| Train      | 70%        |
| Validation | 15%        |
| Test       | 15%        |

### Annotation tools

- [LabelImg](https://github.com/HumanSignal/labelImg) – desktop, YOLO format
- [Roboflow](https://roboflow.com) – web-based, exports YOLO format
- [CVAT](https://www.cvat.ai) – advanced, supports many formats

### Updating `data.yaml`

Edit `data.yaml` to point to your dataset path:

```yaml
path: ./data
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  - license_plate
```

---

## Training

All training scripts accept the same core arguments:

| Argument       | Default                | Description                            |
|----------------|------------------------|----------------------------------------|
| `--data`       | `data.yaml`            | Dataset config YAML                    |
| `--model`      | version-specific       | Pre-trained weights                    |
| `--img-size`   | `640`                  | Input image size (pixels)              |
| `--batch-size` | `16`                   | Batch size (reduce if OOM)             |
| `--epochs`     | `100`                  | Number of training epochs              |
| `--device`     | `0`                    | CUDA device index or `cpu`             |
| `--patience`   | `20`                   | Early-stopping patience                |

### Tips for license plates

- Use `--img-size 1280` for better detection of small/distant plates.
- Start with `yolov8s.pt` or `yolov5s.pt` and scale up as needed.
- If CUDA out-of-memory, halve `--batch-size` or switch to a smaller model.
- Use at least **300 images** per lighting condition for robust results.

---

## Evaluation

### Single model

```bash
python evaluation/evaluate.py \
  --weights runs/train/yolov8_license_plate/weights/best.pt \
  --data    data.yaml \
  --split   val \
  --device  0
```

### Compare multiple models

```bash
python evaluation/compare_models.py \
  --weights runs/train/yolov5_license_plate/weights/best.pt \
            runs/train/yolov8_license_plate/weights/best.pt \
            runs/train/yolov10_license_plate/weights/best.pt \
  --labels  YOLOv5 YOLOv8 YOLOv10 \
  --data    data.yaml \
  --output-chart model_comparison.png
```

---

## Inference & Prediction

### Single image

```bash
python inference/predict_image.py \
  --weights runs/train/yolov8_license_plate/weights/best.pt \
  --source  path/to/image.jpg \
  --conf    0.5
```

### Batch (directory of images)

```bash
python inference/predict_batch.py \
  --weights runs/train/yolov8_license_plate/weights/best.pt \
  --source  path/to/images/ \
  --conf    0.5
```

A CSV summary of detections is saved alongside the annotated images.

### Video file

```bash
python inference/predict_video.py \
  --weights runs/train/yolov8_license_plate/weights/best.pt \
  --source  path/to/video.mp4 \
  --conf    0.5
```

### Real-time webcam / RTSP stream

```bash
# Webcam (device index 0)
python inference/predict_realtime.py \
  --weights runs/train/yolov8_license_plate/weights/best.pt \
  --source  0

# RTSP stream
python inference/predict_realtime.py \
  --weights runs/train/yolov8_license_plate/weights/best.pt \
  --source  rtsp://user:pass@ip:port/stream
```

Press **`q`** to quit the real-time window.

---

## Docker

### Build GPU image

```bash
docker build -f docker/Dockerfile -t yolo-license-plate:gpu .
```

### Build CPU image

```bash
docker build -f docker/Dockerfile.cpu -t yolo-license-plate:cpu .
```

### Run with Docker Compose

```bash
# GPU training (YOLOv8 by default)
docker compose -f docker/docker-compose.yml up train-gpu

# CPU training
docker compose -f docker/docker-compose.yml up train-cpu

# Evaluation
docker compose -f docker/docker-compose.yml up evaluate

# Batch inference
docker compose -f docker/docker-compose.yml up inference
```

### Run a custom training script

```bash
docker compose -f docker/docker-compose.yml run train-gpu \
  python training/train_yolov5.py \
    --data data.yaml \
    --weights yolov5s.pt \
    --epochs 100 \
    --device 0
```

---

## Augmentation Configuration

Edit `configs/augmentation.yaml` to tune augmentation parameters for your
dataset. Key settings for license plates:

| Parameter     | Description                              | Recommended |
|---------------|------------------------------------------|-------------|
| `degrees`     | Random rotation range                    | `5.0`       |
| `perspective` | Perspective distortion                   | `0.001`     |
| `fliplr`      | Horizontal flip probability              | `0.5`       |
| `mosaic`      | Mosaic augmentation probability          | `1.0`       |
| `blur`        | Gaussian blur (simulates motion/focus)   | `0.01`      |

---

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YOLOv5](https://github.com/ultralytics/yolov5)

# debugging a training
docker commit {id} temp_debug_image
docker run -it --entrypoint /bin/sh temp_debug_image
docker image rm temp_debug_image:latest 
ocker container prune

# getting files from container
docker cp temp_debug_image:/data.log ./data.log

# datasets