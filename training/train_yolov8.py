"""
YOLOv8 Fine-tuning Script for License Plate Detection
======================================================

Hardware Requirements:
  Minimum : NVIDIA GPU with 6 GB VRAM (e.g. GTX 1060 / RTX 2060)
  Recommended: NVIDIA GPU with 8–16 GB VRAM (e.g. RTX 3070 / RTX 3080)
  RAM      : 16 GB system RAM
  Storage  : 20 GB free disk space

Installation:
    pip install ultralytics
"""

from ultralytics import YOLO


def train(
    data: str = "data.yaml",
    model: str = "yolov8s.pt",
    img_size: int = 640,
    batch_size: int = 16,
    epochs: int = 100,
    device: str = "0",
    project: str = "runs/train",
    name: str = "yolov8_license_plate",
    patience: int = 20,
    augment: bool = True,
    workers: int = 8,
):
    """
    Fine-tune YOLOv8 on a license plate dataset.

    Args:
        data: Path to the dataset YAML file.
        model: Pre-trained model to start from.
               Options: yolov8n.pt, yolov8s.pt, yolov8m.pt,
                        yolov8l.pt, yolov8x.pt
        img_size: Input image size (pixels).
        batch_size: Training batch size (reduce if OOM, or set -1 for auto).
        epochs: Number of training epochs.
        device: CUDA device index or 'cpu'.
        project: Output directory for training runs.
        name: Name for this training run.
        patience: Early-stopping patience (epochs without improvement).
        augment: Enable built-in Ultralytics augmentations.
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
        augment=augment,
        workers=workers,
        exist_ok=True,
        # License-plate-specific settings
        close_mosaic=10,      # disable mosaic augmentation last N epochs
        degrees=5.0,          # rotation augmentation
        perspective=0.001,    # perspective augmentation
        fliplr=0.5,           # horizontal flip probability
        mosaic=1.0,           # mosaic augmentation probability
    )

    print(f"\nTraining complete. Results saved to: {results.save_dir}")
    return results


if __name__ == "__main__":
    train()
