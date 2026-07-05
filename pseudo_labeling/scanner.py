"""Dataset scanning: enumerate images and label files from a unified data.yaml.

This is the effectful bridge between the on-disk dataset layout and the pure analysis
components. It reads the unified ``data.yaml`` (its ``train`` / ``val`` / ``test`` image
directory lists), derives each directory's dataset id and canonical split, and enumerates
the image files (and their mirrored label paths) within.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

from .paths import image_to_label_path

#: Image extensions recognised when enumerating dataset images.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class ImageRef:
    """A single dataset image and its mirrored (original) label path."""

    dataset_id: str
    split: str  # canonical: train | val | test
    image_path: str
    label_path: str


@dataclass
class DatasetScan:
    """Enumerated images and label files for one dataset, grouped by canonical split."""

    dataset_id: str
    images_by_split: dict[str, list[ImageRef]] = field(default_factory=dict)
    label_files_by_split: dict[str, list[str]] = field(default_factory=dict)


_SPLIT_KEYS = {"train": "train", "val": "val", "test": "test"}


def _dataset_id_and_split(rel_dir: str) -> tuple[str, str]:
    """Derive (dataset_id, split_folder) from a data.yaml image dir like ``ds/images/train``."""
    parts = rel_dir.replace("\\", "/").strip("/").split("/")
    if "images" in parts:
        idx = parts.index("images")
        dataset_id = "/".join(parts[:idx]) or (parts[0] if parts else rel_dir)
        split_folder = parts[idx + 1] if idx + 1 < len(parts) else ""
    else:
        dataset_id = parts[0] if parts else rel_dir
        split_folder = parts[-1]
    return dataset_id, split_folder


def scan_datasets(data_yaml_path: str, datasets_root: str | None = None) -> list[DatasetScan]:
    """Scan every dataset referenced by ``data_yaml_path`` into :class:`DatasetScan` records.

    ``datasets_root`` defaults to the ``path`` key in the data.yaml (or ``./datasets``).
    """
    with open(data_yaml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    root = datasets_root or data.get("path", "./datasets")
    scans: dict[str, DatasetScan] = {}

    for split_key in _SPLIT_KEYS:
        entries = data.get(split_key, [])
        if isinstance(entries, str):
            entries = [entries]
        for rel_dir in entries:
            dataset_id, _folder = _dataset_id_and_split(rel_dir)
            abs_dir = os.path.join(root, rel_dir)
            scan = scans.setdefault(dataset_id, DatasetScan(dataset_id=dataset_id))
            images = _enumerate_images(abs_dir, dataset_id, split_key)
            scan.images_by_split.setdefault(split_key, []).extend(images)
            scan.label_files_by_split.setdefault(split_key, []).extend(
                ref.label_path for ref in images if os.path.isfile(ref.label_path)
            )

    return list(scans.values())


def _enumerate_images(abs_dir: str, dataset_id: str, split: str) -> list[ImageRef]:
    """Recursively enumerate image files under ``abs_dir`` as :class:`ImageRef` records."""
    refs: list[ImageRef] = []
    if not os.path.isdir(abs_dir):
        return refs
    for current, _dirs, files in os.walk(abs_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                image_path = os.path.join(current, name)
                try:
                    label_path = image_to_label_path(image_path)
                except ValueError:
                    continue
                refs.append(
                    ImageRef(
                        dataset_id=dataset_id,
                        split=split,
                        image_path=image_path,
                        label_path=label_path,
                    )
                )
    return refs
