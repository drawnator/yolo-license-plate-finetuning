"""
Video Inference / Prediction Script
=====================================

Runs a trained YOLO model on a video file and saves an annotated output video.

Usage:
    python inference/predict_video.py \
        --weights runs/train/yolov8_license_plate/weights/best.pt \
        --source  path/to/video.mp4 \
        --conf    0.5 \
        --device  0
"""

import argparse
from pathlib import Path


def predict_video(
    weights: str,
    source: str,
    conf: float = 0.5,
    iou: float = 0.45,
    img_size: int = 640,
    device: str = "0",
    output_dir: str = "runs/predict_video",
    save_txt: bool = False,
    save_crop: bool = False,
    show: bool = False,
):
    """
    Run inference on a video file.

    Args:
        weights:    Path to trained model weights.
        source:     Path to input video file (mp4, avi, etc.)
        conf:       Confidence threshold (0–1).
        iou:        IoU threshold for NMS (0–1).
        img_size:   Inference image size (pixels).
        device:     CUDA device index or 'cpu'.
        output_dir: Directory to save annotated output video.
        save_txt:   Save per-frame detection results as .txt files.
        save_crop:  Save cropped detection regions.
        show:       Display the video in a window while processing.
    """
    from ultralytics import YOLO

    model = YOLO(weights)

    print(f"Processing video: {source}")
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name="video",
        save=True,
        save_txt=save_txt,
        save_crop=save_crop,
        show=show,
        stream=True,
        exist_ok=True,
    )

    total_frames = 0
    total_detections = 0
    for r in results:
        total_frames += 1
        if r.boxes is not None:
            total_detections += len(r.boxes)
        if total_frames % 100 == 0:
            print(f"  Processed {total_frames} frames…")

    out_path = Path(output_dir) / "video"
    print(f"\nVideo prediction complete.")
    print(f"  Frames processed  : {total_frames}")
    print(f"  Total detections  : {total_detections}")
    print(f"  Annotated video   : {out_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO license plate video prediction")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Path to input video file")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", default="runs/predict_video")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-crop", action="store_true")
    parser.add_argument("--show", action="store_true", help="Display video while processing")
    args = parser.parse_args()

    predict_video(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        img_size=args.img_size,
        device=args.device,
        output_dir=args.output_dir,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
        show=args.show,
    )


if __name__ == "__main__":
    main()
