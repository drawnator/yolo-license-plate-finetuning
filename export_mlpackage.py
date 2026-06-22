"""Export YOLO model to CoreML .mlpackage and log it to MLflow."""

import os
from pathlib import Path

try:
    import mlflow
except ImportError:
    mlflow = None

from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "runs/detect/runs/train/yolov26_license_plate/weights/best.pt"
EXPERIMENT_NAME = "yolo-license-plate-export"


def export_and_log(model_path: str = MODEL_PATH):
    model = YOLO(model_path)

    # Export to CoreML — returns the path to the .mlpackage directory
    exported_path = model.export(format="coreml", nms=True)
    exported_path = Path(exported_path)
    print(f"Exported to: {exported_path}")

    # Log to MLflow as an artifact
    if mlflow is None:
        print("[mlflow] not installed, skipping artifact logging")
        return

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"export-{exported_path.stem}"):
        mlflow.log_param("source_model", model_path)
        mlflow.log_param("export_format", "coreml")
        mlflow.log_param("nms", True)

        # .mlpackage is a directory — log_artifact handles both files and dirs
        mlflow.log_artifact(str(exported_path))
        print(f"Logged {exported_path.name} to MLflow experiment '{EXPERIMENT_NAME}'")


if __name__ == "__main__":
    export_and_log()
