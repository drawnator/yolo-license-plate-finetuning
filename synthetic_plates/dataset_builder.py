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

Backgrounds that already have YOLO labels (real photos taken from a dataset) keep
them: the original boxes are written next to the synthetic plate boxes, except the
ones a pasted plate covers. Without this, real plates in the background would be
trained as "not a plate".
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

from synthetic_plates.overlay import overlay_on_background, OverlayParams, random_overlay_params
from synthetic_plates.plate_types import PLATE_HEIGHT, PLATE_WIDTH, PlateType, random_plate_type

logger = logging.getLogger(__name__)

Label = Tuple[int, float, float, float, float]

#: An existing box is dropped when a pasted plate hides more than this share of it.
OCCLUSION_DROP_FRACTION = 0.5


def find_background_label(image_path: Path) -> Path | None:
    """Return the YOLO label file of a background image, or ``None`` if it has none.

    Looks in the usual places: ``.../images/x.jpg`` -> ``.../labels/x.txt``, then a
    ``labels/`` folder next to the image, then a ``.txt`` beside the image.
    """
    name = image_path.stem + ".txt"
    candidates = []
    parts = list(image_path.parts)
    for idx in range(len(parts) - 2, -1, -1):
        if parts[idx] == "images":
            candidates.append(Path(*parts[:idx], "labels", *parts[idx + 1:-1], name))
            break
    candidates.append(image_path.parent / "labels" / name)
    candidates.append(image_path.with_suffix(".txt"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_yolo_labels(label_path: Path) -> List[Label]:
    """Read a YOLO label file as boxes. Polygon rows are reduced to their bounding box."""
    labels: List[Label] = []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            cls_id = int(float(fields[0]))
            values = [float(v) for v in fields[1:]]
        except ValueError:
            continue
        if len(values) == 4:
            xc, yc, bw, bh = values
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs, ys = values[0::2], values[1::2]
            xc, yc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        else:
            continue
        if bw > 0 and bh > 0:
            labels.append((cls_id, xc, yc, bw, bh))
    return labels


def covered_fraction(box: Label, by: Label) -> float:
    """Share of ``box``'s area that lies under ``by`` (both normalized xywh)."""
    _, xc, yc, bw, bh = box
    _, oxc, oyc, obw, obh = by
    iw = min(xc + bw / 2, oxc + obw / 2) - max(xc - bw / 2, oxc - obw / 2)
    ih = min(yc + bh / 2, oyc + obh / 2) - max(yc - bh / 2, oyc - obh / 2)
    if iw <= 0 or ih <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    return (iw * ih) / (bw * bh)

# ── Dataset builder ──────────────────────────────────────────────────


class SyntheticDatasetBuilder:
    """Generate a YOLO dataset by overlaying synthetic plates on background images."""

    def __init__(
        self,
        background_dir: str | None,
        output_dir: str,
        plates_per_image: int = 1,
        max_plates_per_image: int = 3,
        image_count: int = 1000,
        val_split: float = 0.1,
        test_split: float = 0.05,
        seed: int | None = None,
        class_names: List[str] | None = None,
        background_sources: Dict[str, List[str]] | None = None,
    ):
        """
        Args:
            background_dir: Directory containing background images (.jpg, .png),
                searched recursively and used for every split. May be ``None`` when
                ``background_sources`` is given.
            output_dir: Where to write the YOLO dataset.
            plates_per_image: Minimum plates to overlay on each image.
            max_plates_per_image: Maximum plates (random between min and max).
            image_count: Total number of images to generate.
            val_split: Fraction of images reserved for validation.
            test_split: Fraction of images reserved for testing.
            seed: Random seed for reproducibility.
            class_names: Class names for ``data.yaml``. Class 0 must be the plate.
                Give the full list when the backgrounds carry labels of other classes.
            background_sources: ``{"train": [dir, ...], "val": [...], "test": [...]}``.
                Each split draws its backgrounds only from its own directories, so a
                photo never appears in two splits. Within a split every directory
                (dataset) is picked equally often, whatever its size. A split left
                out falls back to the ``train`` directories.
        """
        self.class_names = list(class_names) if class_names else ["plate"]
        self._kept_labels = 0
        self._hidden_labels = 0
        self._max_class_id = 0
        self.background_dir = Path(background_dir) if background_dir else None
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

        # Collect backgrounds: per split, one list of images per source directory.
        if not background_sources:
            if self.background_dir is None:
                raise ValueError("Give background_dir or background_sources")
            background_sources = {"train": [str(self.background_dir)]}
        self._groups: Dict[str, List[List[Path]]] = {}
        for split_name, dirs in background_sources.items():
            groups = []
            for d in dirs:
                images = self._scan_dir(Path(d))
                logger.info("Backgrounds for %s: %d images in %s", split_name, len(images), d)
                if images:
                    groups.append(images)
            if groups:
                self._groups[split_name] = groups
        if "train" not in self._groups:
            raise FileNotFoundError(
                f"No images found in {background_sources.get('train') or background_dir}. "
                f"Supported formats: .jpg, .jpeg, .png, .bmp"
            )

    @staticmethod
    def _scan_dir(directory: Path) -> List[Path]:
        """Return every image under ``directory`` (recursive), sorted."""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.rglob("*") if p.suffix.lower() in exts and p.is_file())

    def _pick_background(self, split_name: str) -> Path:
        """Pick a source directory uniformly, then one of its images uniformly."""
        groups = self._groups.get(split_name) or self._groups["train"]
        return random.choice(random.choice(groups))

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
        logger.info(
            "  background labels kept: %d  (dropped because a pasted plate covers them: %d)",
            self._kept_labels, self._hidden_labels,
        )
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
            bg_path = self._pick_background(split_name)
            background = cv2.imread(str(bg_path))
            if background is None:
                logger.warning("Could not read background %s, skipping", bg_path)
                continue

            bg_h, bg_w = background.shape[:2]

            # Determine how many plates to overlay
            n_plates = random.randint(self.plates_per_image, self.max_plates_per_image)

            # Start from the labels the background already has (real plates, faces, ...).
            bg_label_path = find_background_label(bg_path)
            labels: List[Label] = read_yolo_labels(bg_label_path) if bg_label_path else []
            originals = set(labels)
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
                    # The new plate hides whatever is under it, original or pasted earlier.
                    visible = [
                        b for b in labels
                        if covered_fraction(b, label) <= OCCLUSION_DROP_FRACTION
                    ]
                    self._hidden_labels += sum(
                        1 for b in labels if b in originals and b not in visible
                    )
                    labels = visible
                    labels.append(label)

            self._kept_labels += sum(1 for b in labels if b in originals)
            if labels:
                self._max_class_id = max(self._max_class_id, max(b[0] for b in labels))

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
        names = list(self.class_names)
        for class_id in range(len(names), self._max_class_id + 1):
            names.append(f"class_{class_id}")  # carried-over class with no name given
        # Relative paths from the YAML's location
        data_yaml = {
            "path": str(self.output_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(names),
            "names": names,
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        return str(yaml_path)


# ── Convenience function ─────────────────────────────────────────────


def build_dataset(
    background_dir: str | None,
    output_dir: str,
    image_count: int = 1000,
    plates_per_image: int = 1,
    max_plates_per_image: int = 3,
    val_split: float = 0.1,
    test_split: float = 0.05,
    seed: int | None = None,
    class_names: List[str] | None = None,
    background_sources: Dict[str, List[str]] | None = None,
) -> str:
    """Build a synthetic YOLO dataset from backgrounds.

    Labels found for a background image are carried over (see module docstring).

    Args:
        background_dir: Directory of background images.
        output_dir: Output directory for the YOLO dataset.
        image_count: Number of images to generate.
        plates_per_image: Minimum plates per image.
        max_plates_per_image: Maximum plates per image.
        val_split: Validation fraction.
        test_split: Test fraction.
        seed: Random seed.
        class_names: Class names for ``data.yaml`` (class 0 = plate).
        background_sources: Per-split background directories (see the builder).

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
        class_names=class_names,
        background_sources=background_sources,
    )
    return builder.build()