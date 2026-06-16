"""Load and validate dataset configuration from config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    name: str
    urls: List[str]
    zip_path: str
    extract_path: str
    # Optional: converter to apply after extraction (e.g. "ufpr_alpr", "rodosol")
    converter: Optional[str] = None
    # Expected internal structure markers (e.g. ["images", "labels"] or ["train", "valid"])
    expected_structure: List[str] = field(default_factory=list)


def load_config(config_path: str = "config.yaml") -> List[DatasetConfig]:
    """Parse config.yaml and return a list of DatasetConfig objects."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    datasets: List[DatasetConfig] = []
    for name, entry in raw.items():
        urls = entry.get("dataset_url", [])
        if isinstance(urls, str):
            urls = [urls]

        zip_path = entry.get("dataset_zip_path", f"datasets/{name}.zip")
        extract_path = entry.get("dataset_extract_path", f"datasets/{name}")
        converter = entry.get("converter", None)
        expected_structure = entry.get("expected_structure", [])

        datasets.append(
            DatasetConfig(
                name=name,
                urls=urls,
                zip_path=zip_path,
                extract_path=extract_path,
                converter=converter,
                expected_structure=expected_structure,
            )
        )

    return datasets
