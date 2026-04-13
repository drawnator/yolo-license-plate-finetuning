"""
Multi-Model Comparison Script for License Plate Detection
==========================================================

Compares evaluation metrics (mAP50, mAP50-95, Precision, Recall, speed)
across multiple trained YOLO models side-by-side and produces a summary
table and a bar chart.

Usage:
    python evaluation/compare_models.py \
        --weights runs/train/yolov5_license_plate/weights/best.pt \
                  runs/train/yolov8_license_plate/weights/best.pt \
                  runs/train/yolov10_license_plate/weights/best.pt \
        --labels  YOLOv5 YOLOv8 YOLOv10 \
        --data    data.yaml \
        --device  0
"""

import argparse
import time
from pathlib import Path


def evaluate_model(weights: str, data: str, img_size: int, device: str) -> dict:
    """Return a metrics dict for a single model using the Ultralytics API."""
    from ultralytics import YOLO

    model = YOLO(weights)

    start = time.time()
    metrics = model.val(data=data, imgsz=img_size, device=device, verbose=False)
    elapsed = time.time() - start

    return {
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
        "eval_time_s": round(elapsed, 2),
    }


def print_table(labels: list, results: list):
    """Pretty-print a comparison table to stdout."""
    headers = ["Model", "mAP50", "mAP50-95", "Precision", "Recall", "Eval Time (s)"]
    col_w = max(len(h) for h in headers + labels) + 2

    def row(cells):
        return "  ".join(str(c).ljust(col_w) for c in cells)

    sep = "-" * (col_w * len(headers) + 2 * (len(headers) - 1))
    print("\n" + sep)
    print(row(headers))
    print(sep)
    for label, r in zip(labels, results):
        print(
            row([
                label,
                f"{r['mAP50']:.4f}",
                f"{r['mAP50-95']:.4f}",
                f"{r['Precision']:.4f}",
                f"{r['Recall']:.4f}",
                r["eval_time_s"],
            ])
        )
    print(sep + "\n")


def save_chart(labels: list, results: list, output: str):
    """Save a grouped bar chart comparing the key metrics."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed – skipping chart generation.")
        return

    metrics = ["mAP50", "mAP50-95", "Precision", "Recall"]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("License Plate Detection – Model Comparison")
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Chart saved to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple YOLO models")
    parser.add_argument(
        "--weights", nargs="+", required=True, help="Paths to model weight files"
    )
    parser.add_argument(
        "--labels", nargs="+", help="Human-readable labels for each model"
    )
    parser.add_argument("--data", default="data.yaml", help="Path to dataset YAML")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="0", help="CUDA device or 'cpu'")
    parser.add_argument(
        "--output-chart",
        default="model_comparison.png",
        help="Output path for comparison chart",
    )
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [Path(w).parts[-4] for w in args.weights]

    if len(args.labels) != len(args.weights):
        raise ValueError("--labels count must match --weights count")

    results = []
    for label, weights in zip(args.labels, args.weights):
        print(f"Evaluating {label} …")
        results.append(evaluate_model(weights, args.data, args.img_size, args.device))

    print_table(args.labels, results)
    save_chart(args.labels, results, args.output_chart)


if __name__ == "__main__":
    main()
