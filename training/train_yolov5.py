"""
YOLOv5 Fine-tuning Script for License Plate Detection
======================================================

Hardware Requirements:
  Minimum : NVIDIA GPU with 6 GB VRAM (e.g. GTX 1060 / RTX 2060)
  Recommended: NVIDIA GPU with 8–16 GB VRAM (e.g. RTX 3070 / RTX 3080)
  RAM      : 16 GB system RAM
  Storage  : 20 GB free disk space

Installation:
    pip install torch torchvision
    pip install ultralytics  # or: pip install yolov5
"""

import subprocess
import sys
import os


def install_yolov5():
    """Clone and install YOLOv5 dependencies."""
    if not os.path.exists("yolov5"):
        subprocess.run(
            ["git", "clone", "https://github.com/ultralytics/yolov5.git"],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt"],
        check=True,
    )


def train(
    data: str = "data.yaml",
    weights: str = "yolov5s.pt",
    img_size: int = 640,
    batch_size: int = 16,
    epochs: int = 100,
    device: str = "0",
    project: str = "runs/train",
    name: str = "yolov5_license_plate",
    patience: int = 20,
):
    """
    Fine-tune YOLOv5 on a license plate dataset.

    Args:
        data: Path to the dataset YAML file.
        weights: Pre-trained weights to start from.
                 Options: yolov5n.pt, yolov5s.pt, yolov5m.pt,
                          yolov5l.pt, yolov5x.pt
        img_size: Input image size (pixels).
        batch_size: Training batch size (reduce if OOM).
        epochs: Number of training epochs.
        device: CUDA device index or 'cpu'.
        project: Output directory for training runs.
        name: Name for this training run.
        patience: Early-stopping patience (epochs without improvement).
    """
    install_yolov5()

    cmd = [
        sys.executable,
        "yolov5/train.py",
        "--data", data,
        "--weights", weights,
        "--img", str(img_size),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--device", device,
        "--project", project,
        "--name", name,
        "--patience", str(patience),
        "--cache",
        "--exist-ok",
    ]

    print("Starting YOLOv5 training with command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    train()
