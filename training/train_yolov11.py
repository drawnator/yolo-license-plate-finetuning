"""
YOLOv11 Fine-tuning Script for License Plate Detection
=======================================================

Hardware Requirements:
  Minimum : NVIDIA GPU with 8 GB VRAM (e.g. RTX 3060 / RTX 3070)
  Recommended: NVIDIA GPU with 16–24 GB VRAM (e.g. RTX 3090 / RTX 4090)
  RAM      : 32 GB system RAM
  Storage  : 30 GB free disk space

Installation:
    pip install ultralytics>=8.3.0
"""

from ultralytics import YOLO


def train(
    data: str = "data.yaml",
    model: str = "yolo11s.pt",
    img_size: int = 640,
    batch_size: int = 16,
    epochs: int = 100,
    device: str = "0",
    project: str = "runs/train",
    name: str = "yolov11_license_plate",
    patience: int = 20,
    workers: int = 8,
):
    """
    Fine-tune YOLOv11 (YOLO11) on a license plate dataset.

    YOLO11 introduces an improved backbone and neck architecture with
    enhanced feature extraction and a reduced parameter count compared
    to YOLOv8 at equivalent accuracy tiers.

    Args:
        data: Path to the dataset YAML file.
        model: Pre-trained model to start from.
               Options: yolo11n.pt, yolo11s.pt, yolo11m.pt,
                        yolo11l.pt, yolo11x.pt
        img_size: Input image size (pixels).
        batch_size: Training batch size (reduce if OOM, or set -1 for auto).
        epochs: Number of training epochs.
        device: CUDA device index or 'cpu'.
        project: Output directory for training runs.
        name: Name for this training run.
        patience: Early-stopping patience (epochs without improvement).
        workers: Number of data-loading workers.
    """
    yolo = YOLO(model)

    results = yolo.train(
        data=data,
        imgsz=img_size,
        batch=batch_size,
        epochs=epochs,
        device=device,
        project=project,
        name=name,
        patience=patience,
        workers=workers,
        exist_ok=True,
        # License-plate-specific settings
        close_mosaic=10,
        degrees=5.0,
        perspective=0.001,
        fliplr=0.5,
        mosaic=1.0,
    )

    print(f"\nTraining complete. Results saved to: {results.save_dir}")
    return results


if __name__ == "__main__":
    train()
