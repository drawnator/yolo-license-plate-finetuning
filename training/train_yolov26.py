"""
YOLOv26n Fine-tuning Script for License Plate Detection
=======================================================

Hardware Requirements:
    Minimum : NVIDIA GPU with 10 GB VRAM (e.g. RTX 3080 / RTX 4070)
    Recommended: NVIDIA GPU with 24 GB VRAM (e.g. RTX 4090 / A6000)
    RAM      : 32–64 GB system RAM
    Storage  : 30 GB free disk space

Installation:
    pip install ultralytics>=8.3.0
"""

from ultralytics import YOLO


def train(
    data: str = r"app\datasets\brazil_yolo12\data.yaml",
    # data: str = r"datasets\UFPR-ALPR dataset\data.yaml",
    model: str = "	yolo26n.pt",
    # img_size: int = 1920,
    batch_size: int = 16,
    epochs: int = 100,
    device: str = "0",
    project: str = "runs/train",
    name: str = "yolov26_license_plate",
    patience: int = 20,
    workers: int = 8,
):
    """
    Fine-tune YOLOv26 on a license plate dataset.

    YOLOv12 integrates attention-centric architecture improvements
    (area attention) enabling better global context capture while
    maintaining real-time inference speeds.

    Args:
        data: Path to the dataset YAML file.
        model: Pre-trained model to start from.
               Options: yolov12n.pt, yolov12s.pt, yolov12m.pt,
                        yolov12l.pt, yolov12x.pt
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
    print("loaded model")
    results = yolo.train(
        data=data,
        # imgsz=img_size,
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
