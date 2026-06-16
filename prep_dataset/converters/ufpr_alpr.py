"""Converter for the UFPR-ALPR dataset to YOLO format.

The raw dataset has annotation .txt files alongside .png images in a
nested folder structure (training/validation/testing -> vehicle type -> images).

Each annotation file has the format:
  Line 1: type: car/motorcycle
  Line 2: plate position (x, y, w, h)  -- but we use the corners from line 7+
  ...
  Line 7+: plate corners

This converter creates parallel images/ and labels/ directories preserving
the train/validation/testing split structure.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DIMENSIONS = [1_920, 1_080]


def convert_ufpr_alpr(extract_path: str) -> None:
    """Convert UFPR-ALPR dataset in-place to YOLO format with images/ and labels/ dirs."""
    root = Path(extract_path)

    # Skip if already converted (labels directory exists with files)
    labels_dir = root / "labels"
    if labels_dir.exists() and any(labels_dir.rglob("*.txt")):
        logger.info("[ufpr_alpr] already converted, skipping")
        return

    converted = 0
    skipped = 0

    for txt_file in root.rglob("*.txt"):
        if txt_file.name == "README.txt":
            continue

        # Determine relative path from extract root
        rel = txt_file.relative_to(root)

        try:
            with open(txt_file) as f:
                lines = f.readlines()
            id_type, vehicle_pos, plate_pos = _get_data(lines)
        except (IndexError, ValueError) as e:
            logger.debug("[ufpr_alpr] skipping %s: %s", txt_file, e)
            skipped += 1
            continue

        # Map vehicle type to class id
        class_id = 2 if id_type == "car" else 3

        vehicle_bbox = _convert_vehicle(vehicle_pos)
        plate_bbox = _convert_plate(plate_pos)

        # Create label file under labels/ subtree
        label_path = root / "labels" / rel
        label_path.parent.mkdir(parents=True, exist_ok=True)

        with open(label_path, "w") as f:
            f.write(f"0 {plate_bbox[0]} {plate_bbox[1]} {plate_bbox[2]} {plate_bbox[3]}\n")
            f.write(f"{class_id} {vehicle_bbox[0]} {vehicle_bbox[1]} {vehicle_bbox[2]} {vehicle_bbox[3]}")

        converted += 1

    # Move image files to images/ subtree
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    moved = 0
    for img_file in list(root.rglob("*")):
        if img_file.suffix.lower() not in image_exts:
            continue
        # Skip files already under images/
        try:
            img_file.relative_to(root / "images")
            continue
        except ValueError:
            pass

        rel = img_file.relative_to(root)
        dest = root / "images" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_file), str(dest))
        moved += 1

    logger.info(
        "[ufpr_alpr] converted=%d skipped=%d images_moved=%d",
        converted, skipped, moved,
    )


def _get_data(lines):
    """Parse UFPR-ALPR annotation lines."""
    id_type = lines[2].split(":")[-1].strip()
    vehicle_pos = list(map(int, lines[1].split(":")[-1].strip().split()))
    plate_pos = lines[7].split(":")[-1].strip().split()
    plate_pos = [list(map(int, i.split(","))) for i in plate_pos]
    return id_type, vehicle_pos, plate_pos


def _convert_vehicle(vehicle_pos):
    """Convert vehicle bounding box to YOLO format (cx, cy, w, h) normalized."""
    x = vehicle_pos[0] / DIMENSIONS[0]
    y = vehicle_pos[1] / DIMENSIONS[1]
    w = vehicle_pos[2] / DIMENSIONS[0]
    h = vehicle_pos[3] / DIMENSIONS[1]
    return [x + w / 2, y + h / 2, w, h]


def _convert_plate(plate_pos):
    """Convert plate corner points to YOLO bounding box (cx, cy, w, h) normalized."""
    xs = [p[0] for p in plate_pos]
    ys = [p[1] for p in plate_pos]
    x = sum(xs) / 4 / DIMENSIONS[0]
    y = sum(ys) / 4 / DIMENSIONS[1]
    w = (max(xs) - min(xs)) / DIMENSIONS[0]
    h = (max(ys) - min(ys)) / DIMENSIONS[1]
    return [x, y, w, h]
