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

import logging
import os
import shutil
import sys
from pathlib import Path

try:
    import mlflow
except ImportError:
    mlflow = None

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def _check_environment():
    """Validate environment before starting training."""
    if not Path("datasets").exists():
        raise RuntimeError(
            "No datasets/ directory. Run `python -m prep_dataset` first."
        )

    # Disk space warning
    free_gb = shutil.disk_usage(".").free / (1024**3)
    if free_gb < 5:
        logger.warning("Less than 5GB free disk space — training may fail")

    # CUDA availability
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA is available")
        else:
            logger.warning("CUDA is not available — training will use CPU")
    except ImportError:
        logger.warning("torch not installed, assuming CPU-only")


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

    Returns:
        Ultralytics training results.
    """
    # --- Input validation ---
    data_path = Path(data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data}")

    if not model.lower().endswith(".pt"):
        raise ValueError(
            f"Expected a .pt model file, got '{model}'. "
            f"Valid examples: yolo26n.pt, yolo26s.pt, yolo26m.pt"
        )

    if not Path(model).exists():
        logger.info(f"[info] {model} not found locally, will be downloaded by ultralytics")

    if not (1 <= batch_size <= 256):
        raise ValueError(f"batch_size must be between 1 and 256, got {batch_size}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if patience < 0:
        raise ValueError(f"patience must be non‑negative, got {patience}")
    if workers < 0:
        raise ValueError(f"workers must be non‑negative, got {workers}")

    # Normalize device string
    if device != "cpu":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(f"device={device} but CUDA unavailable, falling back to cpu")
                device = "cpu"
        except ImportError:
            device = "cpu"

    # Idempotent output directory creation
    project_path = Path(project).resolve()
    project_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training outputs will be saved to: {project_path}")

    # --- Start training ---
    logger.info(f"Loading model {model}...")
    yolo = YOLO(model)

    try:
        results = yolo.train(
            data=str(data_path),
            batch=batch_size,
            epochs=epochs,
            device=device,
            project=str(project_path),
            name=name,
            patience=patience,
            workers=workers,
            exist_ok=True,
            # License‑plate‑specific augmentation
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
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user. Partial checkpoint may be in: %s", project_path)
        sys.exit(1)
    except Exception as e:
        logger.error("Training failed: %s", e)
        raise

    logger.info(f"Training complete. Results saved to: {results.save_dir}")

    # --- Export best model to CoreML and log to MLflow (optional) ---
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        _export_and_log(best_weights)
    else:
        logger.warning("best.pt not found after training; skipping export")

    return results


def _export_and_log(weights_path: Path):
    """Export best weights to .mlpackage and log the artifact to MLflow (if available)."""
    logger.info(f"Exporting {weights_path.name} to CoreML...")
    try:
        model = YOLO(str(weights_path))
        exported_path = Path(model.export(format="coreml", nms=True))
        logger.info(f"Exported to: {exported_path}")
    except Exception as e:
        logger.error("Export to CoreML failed: %s", e)
        return

    if mlflow is None:
        logger.warning("mlflow not installed, skipping artifact logging")
        return

    try:
        mlflow.set_experiment("yolo-license-plate-export")
        with mlflow.start_run(run_name=f"export-{exported_path.stem}"):
            mlflow.log_param("source_weights", str(weights_path))
            mlflow.log_param("export_format", "coreml")
            mlflow.log_param("nms", True)
            mlflow.log_artifact(str(exported_path))
            logger.info(f"Logged {exported_path.name} to MLflow")
    except Exception as e:
        # Export succeeded — don't raise just because logging failed
        logger.error("MLflow logging failed: %s", e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    _check_environment()
    train()
