"""
Batch Inference / Prediction Script
=====================================

Runs a trained YOLO model on all images in a directory (or glob pattern)
and saves annotated results plus a CSV summary.

Usage:
    python inference/predict_batch.py \
        --weights runs/train/yolov8_license_plate/weights/best.pt \
        --source  path/to/images/ \
        --conf    0.5 \
        --device  0
"""

import argparse
import csv
from pathlib import Path


def predict_batch(
    weights: str,
    source: str,
    conf: float = 0.5,
    iou: float = 0.45,
    img_size: int = 640,
    device: str = "0",
    output_dir: str = "runs/predict_batch",
    batch_size: int = 32,
    save_txt: bool = False,
    save_crop: bool = False,
):
    """
    Run batch inference on a folder of images.

    Args:
        weights:    Path to trained model weights.
        source:     Directory containing images.
        conf:       Confidence threshold (0–1).
        iou:        IoU threshold for NMS (0–1).
        img_size:   Inference image size (pixels).
        device:     CUDA device index or 'cpu'.
        output_dir: Directory to save annotated results and CSV summary.
        batch_size: Number of images per inference batch.
        save_txt:   Save detection results as .txt files.
        save_crop:  Save cropped detection regions.
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name="batch",
        save=True,
        save_txt=save_txt,
        save_crop=save_crop,
        stream=True,
        exist_ok=True,
    )

    output_dir_path = Path(output_dir) / "batch"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir_path / "detections.csv"

    total_images = 0
    total_detections = 0

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "num_detections", "confidences", "bboxes"])

        for r in results:
            total_images += 1
            boxes = r.boxes
            num_det = len(boxes) if boxes is not None else 0
            total_detections += num_det

            confs = (
                [round(float(c), 3) for c in boxes.conf.tolist()]
                if boxes is not None and len(boxes) > 0
                else []
            )
            bboxes = (
                [[round(v, 1) for v in b] for b in boxes.xyxy.tolist()]
                if boxes is not None and len(boxes) > 0
                else []
            )
            writer.writerow([r.path, num_det, confs, bboxes])

    print(f"\nBatch prediction complete.")
    print(f"  Images processed : {total_images}")
    print(f"  Total detections : {total_detections}")
    print(f"  Results saved to : {output_dir_path}")
    print(f"  CSV summary      : {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO license plate batch prediction")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Directory with images")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", default="runs/predict_batch")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-crop", action="store_true")
    args = parser.parse_args()

    predict_batch(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        img_size=args.img_size,
        device=args.device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
    )


if __name__ == "__main__":
    main()
