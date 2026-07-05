"""Audit_Report builder and persistence (Req 7, 8).

The Audit_Report lists every proposed pseudo-label (image path, class id, box, confidence)
and summary counts (per class per dataset, and images that received zero accepted labels
per dataset and per split). It is always produced regardless of merge mode; in
Auto_Merge_Mode it is informational only.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field

from .models import Diagnostic, PseudoLabel


@dataclass(frozen=True)
class AuditEntry:
    """A single proposed pseudo-label recorded for review (Req 8.1)."""

    image_path: str
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float

    @classmethod
    def from_pseudo_label(cls, pl: PseudoLabel) -> "AuditEntry":
        return cls(
            image_path=pl.image_path,
            class_id=pl.class_id,
            x_center=pl.box.x_center,
            y_center=pl.box.y_center,
            width=pl.box.width,
            height=pl.box.height,
            confidence=pl.confidence,
        )


@dataclass
class AuditReport:
    """Machine-readable record of proposed pseudo-labels and run summary (Req 7.6, 8.1, 8.5)."""

    entries: list[AuditEntry] = field(default_factory=list)
    # counts[dataset_id][class_id] -> number proposed (Req 8.5)
    per_class_per_dataset: dict = field(default_factory=dict)
    # zero_label_images[dataset_id] = {"total": N, "per_split": {split: n}} (Req 7.6)
    zero_label_images: dict = field(default_factory=dict)
    write_diagnostics: list[Diagnostic] = field(default_factory=list)

    def add_proposal(self, pl: PseudoLabel, dataset_id: str) -> None:
        """Record a proposed pseudo-label and bump its per-class-per-dataset count (Req 8.1, 8.5)."""
        self.entries.append(AuditEntry.from_pseudo_label(pl))
        bucket = self.per_class_per_dataset.setdefault(dataset_id, {})
        bucket[pl.class_id] = bucket.get(pl.class_id, 0) + 1

    def record_zero_label_image(self, dataset_id: str, split: str) -> None:
        """Count an image that received zero accepted pseudo-labels (Req 7.6)."""
        bucket = self.zero_label_images.setdefault(dataset_id, {"total": 0, "per_split": {}})
        bucket["total"] += 1
        bucket["per_split"][split] = bucket["per_split"].get(split, 0) + 1

    @property
    def total_proposed(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict:
        return {
            "total_proposed": self.total_proposed,
            "entries": [asdict(e) for e in self.entries],
            "per_class_per_dataset": self.per_class_per_dataset,
            "zero_label_images": self.zero_label_images,
            "write_diagnostics": [asdict(d) for d in self.write_diagnostics],
        }

    def save(self, path: str) -> str:
        """Atomically write the audit report as JSON (Req 8.1)."""
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return path
