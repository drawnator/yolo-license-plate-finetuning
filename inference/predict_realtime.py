"""
Real-Time Webcam / RTSP Stream Inference Script
================================================

Runs a trained YOLO model on a live camera feed or RTSP stream
and displays the annotated output in real time using OpenCV.

Requirements:
    pip install ultralytics opencv-python

Usage:
    # Webcam (device 0)
    python inference/predict_realtime.py \
        --weights runs/train/yolov8_license_plate/weights/best.pt \
        --source  0 \
        --conf    0.5

    # RTSP stream
    python inference/predict_realtime.py \
        --weights runs/train/yolov8_license_plate/weights/best.pt \
        --source  rtsp://username:password@ip:port/stream \
        --conf    0.5
"""

import argparse
import time


def predict_realtime(
    weights: str,
    source: int | str = 0,
    conf: float = 0.5,
    iou: float = 0.45,
    img_size: int = 640,
    device: str = "0",
    show: bool = True,
):
    """
    Run real-time inference on a webcam or RTSP stream.

    Press 'q' to quit.

    Args:
        weights:  Path to trained model weights.
        source:   Camera device index (int) or RTSP/HTTP URL (str).
        conf:     Confidence threshold (0–1).
        iou:      IoU threshold for NMS (0–1).
        img_size: Inference image size (pixels).
        device:   CUDA device index or 'cpu'.
        show:     Display the annotated frame in a window.
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    print(f"Streaming from: {source}  (press 'q' to quit)")
    fps_start = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or cannot read frame.")
            break

        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            device=device,
            verbose=False,
        )

        annotated = results[0].plot()
        frame_count += 1

        # Compute and overlay FPS
        elapsed = time.time() - fps_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if show:
            cv2.imshow("License Plate Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\nSession complete. Avg FPS: {frame_count / (time.time() - fps_start):.1f}")


def main():
    parser = argparse.ArgumentParser(description="YOLO real-time license plate detection")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", default="0", help="Camera index or RTSP URL")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-show", action="store_true", help="Disable display window")
    args = parser.parse_args()

    predict_realtime(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        img_size=args.img_size,
        device=args.device,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
