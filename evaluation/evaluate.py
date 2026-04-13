"""
Model Evaluation Script for License Plate Detection
====================================================

Evaluates a trained YOLO model on the validation/test set and reports:
  - mAP50
  - mAP50-95
  - Precision / Recall
  - Inference speed (ms/image)

Supports: YOLOv8, YOLOv10, YOLOv11, YOLOv12 (Ultralytics API)
          YOLOv5 (via subprocess)

Usage:
    python evaluation/evaluate.py --weights runs/train/yolov8_license_plate/weights/best.pt \
                                   --data data.yaml --device 0
"""

import argparse
import subprocess
import sys
from pathlib import Path


def evaluate_ultralytics(weights: str, data: str, img_size: int, device: str, split: str):
    """Evaluate a model using the Ultralytics API (YOLOv8/v10/v11/v12)."""
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(
        data=data,
        imgsz=img_size,
        device=device,
        split=split,
    )

    print("\n========== Evaluation Results ==========")
    print(f"  Model      : {weights}")
    print(f"  Dataset    : {data}  (split={split})")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")
    print("========================================\n")
    return metrics


def evaluate_yolov5(weights: str, data: str, img_size: int, device: str):
    """Evaluate a YOLOv5 model (requires yolov5/ directory to be present)."""
    yolov5_val = Path("yolov5/val.py")
    if not yolov5_val.exists():
        print("YOLOv5 not found. Cloning repository…")
        subprocess.run(
            ["git", "clone", "https://github.com/ultralytics/yolov5.git"],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt"],
            check=True,
        )

    cmd = [
        sys.executable, str(yolov5_val),
        "--weights", weights,
        "--data", data,
        "--img", str(img_size),
        "--device", device,
        "--verbose",
    ]
    print("Running YOLOv5 validation:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a YOLO license plate model")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    parser.add_argument("--data", default="data.yaml", help="Path to dataset YAML")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="0", help="CUDA device or 'cpu'")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--yolov5",
        action="store_true",
        help="Use YOLOv5 evaluation (subprocess mode)",
    )
    args = parser.parse_args()

    if args.yolov5:
        evaluate_yolov5(args.weights, args.data, args.img_size, args.device)
    else:
        evaluate_ultralytics(args.weights, args.data, args.img_size, args.device, args.split)


if __name__ == "__main__":
    main()
