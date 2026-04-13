"""
Single-Image Inference / Prediction Script
===========================================

Runs a trained YOLO model on a single image and saves the annotated result.

Usage:
    python inference/predict_image.py \
        --weights runs/train/yolov8_license_plate/weights/best.pt \
        --source  path/to/image.jpg \
        --conf    0.5 \
        --device  0
"""

import argparse
from pathlib import Path


def predict(
    weights: str,
    source: str,
    conf: float = 0.5,
    iou: float = 0.45,
    img_size: int = 640,
    device: str = "0",
    output_dir: str = "runs/predict",
    save_txt: bool = False,
    save_crop: bool = False,
):
    """
    Run inference on a single image.

    Args:
        weights:    Path to trained model weights.
        source:     Path to input image.
        conf:       Confidence threshold (0–1).
        iou:        IoU threshold for NMS (0–1).
        img_size:   Inference image size (pixels).
        device:     CUDA device index or 'cpu'.
        output_dir: Directory to save annotated results.
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
        name="image",
        save=True,
        save_txt=save_txt,
        save_crop=save_crop,
        exist_ok=True,
    )

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print("No license plates detected.")
        else:
            print(f"Detected {len(boxes)} license plate(s):")
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                conf_score = float(box.conf[0])
                print(f"  [{i + 1}] bbox={[round(v, 1) for v in xyxy]}  conf={conf_score:.3f}")

    save_path = Path(output_dir) / "image"
    print(f"\nAnnotated image saved to: {save_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="YOLO license plate single-image prediction")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", default="runs/predict")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-crop", action="store_true")
    args = parser.parse_args()

    predict(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        img_size=args.img_size,
        device=args.device,
        output_dir=args.output_dir,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
    )


if __name__ == "__main__":
    main()
