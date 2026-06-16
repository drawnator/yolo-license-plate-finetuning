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

from pathlib import Path

import mlflow
from ultralytics import YOLO


def train(
    data: str = r"./data.yaml",
    model: str = "yolo26s.pt",
    batch_size: int = 16,
    epochs: int = 100,
    device: str = "cpu",
    project: str = "runs/train",
    name: str = "yolov26_license_plate",
    patience: int = 20,
    workers: int = 8,
):
    """
    Fine-tune YOLOv26 on a license plate dataset.

    Args:
        data: Path to the dataset YAML file.
        model: Pre-trained model to start from.
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
        fliplr=0.5,
        mosaic=1.0,
        multi_scale=0.25,
        shear=45,
        perspective=0.001,
        cutmix=0.1,
        mixup=0.1,
    )

    print(f"\nTraining complete. Results saved to: {results.save_dir}")

    # --- Export to CoreML and log to MLflow ---
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        export_and_log(best_weights)

    return results


def export_and_log(weights_path: Path):
    """Export best weights to .mlpackage and log the artifact to MLflow."""
    print(f"\nExporting {weights_path} to CoreML...")
    model = YOLO(str(weights_path))
    exported_path = Path(model.export(format="coreml", nms=True))
    print(f"Exported to: {exported_path}")

    mlflow.set_experiment("yolo-license-plate-export")
    with mlflow.start_run(run_name=f"export-{exported_path.stem}"):
        mlflow.log_param("source_weights", str(weights_path))
        mlflow.log_param("export_format", "coreml")
        mlflow.log_param("nms", True)
        mlflow.log_artifact(str(exported_path))
        print(f"Logged {exported_path.name} to MLflow")


if __name__ == "__main__":
    train()
