"""Build a complete YOLO-format dataset from synthetic plates overlaid on backgrounds.

Produces:
    output_dir/
        images/
            000001.jpg
            000002.jpg
            ...
        labels/
            000001.txt
            000002.txt
            ...
        data.yaml          # dataset descriptor referencing these paths

Each image contains a background with one or more synthetic plates overlaid.
Labels are YOLO-format: ``class_id x_center y_center width height`` (normalized).
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml

from synthetic_plates.overlay import overlay_on_background, OverlayParams, random_overlay_params
from synthetic_plates.plate_types import PLATE_HEIGHT, PLATE_WIDTH, PlateType, random_plate_type

logger = logging.getLogger(__name__)

# ── Dataset builder ──────────────────────────────────────────────────


class SyntheticDatasetBuilder:
    """Generate a YOLO dataset by overlaying synthetic plates on background images."""

    def __init__(
        self,
        background_dir: str,
        output_dir: str,
        plates_per_image: int = 1,
        max_plates_per_image: int = 3,
        image_count: int = 1000,
        val_split: float = 0.1,
        test_split: float = 0.05,
        seed: int | None = None,
    ):
        """
        Args:
            background_dir: Directory containing background images (.jpg, .png).
            output_dir: Where to write the YOLO dataset.
            plates_per_image: Minimum plates to overlay on each image.
            max_plates_per_image: Maximum plates (random between min and max).
            image_count: Total number of images to generate.
            val_split: Fraction of images reserved for validation.
            test_split: Fraction of images reserved for testing.
            seed: Random seed for reproducibility.
        """
        self.background_dir = Path(background_dir)
        self.output_dir = Path(output_dir)
        self.plates_per_image = plates_per_image
        self.max_plates_per_image = max(max_plates_per_image, plates_per_image)
        self.image_count = image_count
        self.val_split = val_split
        self.test_split = test_split
        self.train_split = 1.0 - val_split - test_split

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Collect backgrounds
        self._backgrounds = self._scan_backgrounds()
        if not self._backgrounds:
            raise FileNotFoundError(
                f"No images found in {background_dir}. "
                f"Supported formats: .jpg, .jpeg, .png, .bmp"
            )

    def _scan_backgrounds(self) -> List[Path]:
        """Scan the background directory for image files."""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        files: List[Path] = []
        for ext in exts:
            files.extend(self.background_dir.glob(f"*{ext}"))
            files.extend(self.background_dir.glob(f"*{ext.upper()}"))
        return sorted(files)

    def build(self) -> str:
        """Generate the full dataset. Returns the path to the output ``data.yaml``."""
        # Partition: train / val / test
        n_total = self.image_count
        n_test = max(1, int(n_total * self.test_split))
        n_val = max(1, int(n_total * self.val_split))
        n_train = n_total - n_val - n_test

        splits = [("train", n_train), ("val", n_val), ("test", n_test)]

        for split_name, count in splits:
            if count <= 0:
                continue
            self._generate_split(split_name, count)

        # Write data.yaml
        data_yaml_path = self._write_data_yaml()
        logger.info("Dataset built: %s", self.output_dir)
        logger.info("  train: %d  val: %d  test: %d", n_train, n_val, n_test)
        logger.info("  data.yaml: %s", data_yaml_path)
        return data_yaml_path

    def _generate_split(self, split_name: str, count: int) -> None:
        """Generate images + labels for one dataset split."""
        img_dir = self.output_dir / split_name / "images"
        lbl_dir = self.output_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating %s split: %d images...", split_name, count)

        for i in range(count):
            # Pick a random background
            bg_path = random.choice(self._backgrounds)
            background = cv2.imread(str(bg_path))
            if background is None:
                logger.warning("Could not read background %s, skipping", bg_path)
                continue

            bg_h, bg_w = background.shape[:2]

            # Determine how many plates to overlay
            n_plates = random.randint(self.plates_per_image, self.max_plates_per_image)

            labels: List[Tuple[int, float, float, float, float]] = []
            img = background.copy()

            for _ in range(n_plates):
                plate_type = random_plate_type()
                img, label = overlay_on_background(
                    background=img,
                    plate_type=plate_type,
                    params=random_overlay_params(),
                )

                # Validate the label is reasonable
                if label[3] > 0.001 and label[4] > 0.001:
                    labels.append(label)

            # Save image
            img_name = f"{split_name}_{i:06d}.jpg"
            img_path = img_dir / img_name
            cv2.imwrite(str(img_path), img)

            # Save labels
            lbl_name = f"{split_name}_{i:06d}.txt"
            lbl_path = lbl_dir / lbl_name
            with open(lbl_path, "w") as f:
                for lbl in labels:
                    cls_id, xc, yc, bw, bh = lbl
                    # Clamp to [0, 1]
                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))
                    f.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            if (i + 1) % 100 == 0:
                logger.info("  %s: %d/%d", split_name, i + 1, count)

    def _write_data_yaml(self) -> str:
        """Write the YOLO dataset descriptor YAML."""
        # Relative paths from the YAML's location
        data_yaml = {
            "path": str(self.output_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 1,
            "names": ["plate"],
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        return str(yaml_path)


# ── Convenience function ─────────────────────────────────────────────


def build_dataset(
    background_dir: str,
    output_dir: str,
    image_count: int = 1000,
    plates_per_image: int = 1,
    max_plates_per_image: int = 3,
    val_split: float = 0.1,
    test_split: float = 0.05,
    seed: int | None = None,
) -> str:
    """Build a synthetic YOLO dataset from backgrounds.

    Args:
        background_dir: Directory of background images.
        output_dir: Output directory for the YOLO dataset.
        image_count: Number of images to generate.
        plates_per_image: Minimum plates per image.
        max_plates_per_image: Maximum plates per image.
        val_split: Validation fraction.
        test_split: Test fraction.
        seed: Random seed.

    Returns:
        Path to the generated ``data.yaml``.
    """
    builder = SyntheticDatasetBuilder(
        background_dir=background_dir,
        output_dir=output_dir,
        plates_per_image=plates_per_image,
        max_plates_per_image=max_plates_per_image,
        image_count=image_count,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )
    return builder.build()