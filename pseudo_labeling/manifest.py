"""Run_Manifest model and store (Req 2, 4, 9, 10, 13, 14).

The manifest is the reproducibility record for a run: configuration, seed, teacher model
ids, per-class training datasets, applied thresholds, selected merge mode / output target,
and any produced label set. It is written atomically so a failed save never leaves a
partial manifest.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field


class ManifestStoreError(Exception):
    """Raised when the Run_Manifest cannot be stored (Req 9.6)."""


@dataclass
class RunManifest:
    """Reproducible record of a pseudo-labeling run (Req 9.1, 9.7)."""

    run_id: str
    seed: int
    config: dict = field(default_factory=dict)
    teacher_models: dict = field(default_factory=dict)          # class_id -> weights path
    training_datasets: dict = field(default_factory=dict)       # class_id -> [dataset ids]
    thresholds: dict = field(default_factory=dict)              # applied per-class + default
    merge_mode: str = "review"                                  # Req 9.7
    non_interactive: bool = False                               # Req 9.7
    output_target: str = "in-place"                             # Req 9.7, 13.1
    label_set_id: str | None = None                            # Req 14.3
    label_set_location: str | None = None                      # Req 14.3
    status: str = "running"                                     # running | success | failed

    def to_dict(self) -> dict:
        return asdict(self)


class RunManifestStore:
    """Persist and load :class:`RunManifest` documents under the run directory."""

    def __init__(self, runs_root: str = "runs/pseudo_labeling") -> None:
        self.runs_root = runs_root

    def path_for(self, run_id: str) -> str:
        """Return the manifest JSON path for a run id."""
        return os.path.join(self.runs_root, run_id, "manifest.json")

    def save(self, manifest: RunManifest) -> str:
        """Atomically write a manifest; raise :class:`ManifestStoreError` on failure (Req 9.6)."""
        path = self.path_for(manifest.run_id)
        try:
            parent = os.path.dirname(path) or "."
            os.makedirs(parent, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
            os.replace(tmp, path)
            return path
        except OSError as exc:
            raise ManifestStoreError(
                f"could not store manifest for run {manifest.run_id!r} at {path!r}: {exc}"
            ) from exc

    def load(self, run_id: str) -> RunManifest:
        """Load a manifest by run id."""
        path = self.path_for(run_id)
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return RunManifest(**data)
