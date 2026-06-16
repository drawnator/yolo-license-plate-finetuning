"""Converter for the RodoSol-ALPR dataset to YOLO format.

RodoSol-ALPR annotation files have this shape:
  type: car            (or motorcycle)
  plate: ODE2510
  layout: Brazilian    (or Mercosur)
  corners: x1,y1 x2,y2 x3,y3 x4,y4

Images are 1280x720. The dataset only provides plate corners (no vehicle
bounding box), so we emit a single YOLO label per image: the plate (class 0).
split.txt assigns each image to training / validation / testing.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DIMENSIONS = [1_280, 720]

_SPLIT_MAP = {
    "training": "training",
    "train": "training",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "testing": "testing",
    "test": "testing",
}


def convert_rodosol(extract_path: str) -> None:
    """Convert RodoSol-ALPR dataset in-place to YOLO split structure."""
    root = Path(extract_path)

    split_file = root / "split.txt"
    if not split_file.exists():
        logger.warning("[rodosol] split.txt not found in %s, skipping", extract_path)
        return

    images_root = root / "images"
    labels_root = root / "labels"

    # Skip if already converted
    if labels_root.exists() and any(labels_root.rglob("*.txt")):
        logger.info("[rodosol] already converted, skipping")
        return

    with open(split_file) as f:
        entries = [line.strip() for line in f if line.strip()]

    converted = 0
    skipped = 0

    for entry in entries:
        rel_image, _, split_name = entry.partition(";")
        split_name = _SPLIT_MAP.get(split_name.strip().lower())
        if not split_name:
            skipped += 1
            continue

        # rel_image looks like "./images/cars-br/img_000003.jpg"
        rel_image = rel_image.lstrip("./")
        src_image = root / rel_image
        src_label = src_image.with_suffix(".txt")
        if not src_image.exists() or not src_label.exists():
            skipped += 1
            continue

        # Preserve the subgroup folder (cars-br, cars-me, ...) under the split.
        subgroup = src_image.parent.name
        image_name = src_image.name
        label_name = src_label.with_suffix(".txt").name

        dst_image = images_root / split_name / subgroup / image_name
        dst_label = labels_root / split_name / subgroup / label_name
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(src_label) as fh:
                vehicle_type, corners = _parse_label(fh.readlines())
        except (IndexError, ValueError) as e:
            logger.debug("[rodosol] skipping %s: %s", src_label, e)
            skipped += 1
            continue

        if len(corners) != 4:
            skipped += 1
            continue

        plate_bbox = _corners_to_yolo_bbox(corners)

        with open(dst_label, "w") as fh:
            fh.write(
                f"0 {plate_bbox[0]} {plate_bbox[1]} {plate_bbox[2]} {plate_bbox[3]}\n"
            )

        shutil.move(str(src_image), str(dst_image))
        converted += 1

    logger.info("[rodosol] converted=%d skipped=%d", converted, skipped)


def _parse_label(lines):
    """Parse a RodoSol annotation file into (vehicle_type, corners)."""
    vehicle_type = ""
    corners = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "type":
            vehicle_type = value.lower()
        elif key == "corners":
            corners = [list(map(int, p.split(","))) for p in value.split()]
    return vehicle_type, corners


def _corners_to_yolo_bbox(corners):
    """Convert 4 corner points to YOLO bbox (cx, cy, w, h) normalized."""
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = (x_max - x_min) / DIMENSIONS[0]
    h = (y_max - y_min) / DIMENSIONS[1]
    cx = (x_min + x_max) / 2 / DIMENSIONS[0]
    cy = (y_min + y_max) / 2 / DIMENSIONS[1]
    return [cx, cy, w, h]
