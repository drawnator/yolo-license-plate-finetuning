"""Teacher_Trainer: train class-specialized teachers on class-containing data (Req 2).

Training itself is delegated to the injected :class:`~pseudo_labeling.backends.TeacherBackend`
(the real backend wraps ``training/train_yolov26.py::train``); this module owns the
selection of which datasets may train a teacher for a target class, base-model validation,
manifest recording, and weight persistence bookkeeping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .backends import TeacherBackend
from .coverage import DatasetCoverage
from .models import Diagnostic

#: Diagnostic codes for the trainer's error paths.
NO_DATASET_FOR_CLASS = "NO_DATASET_FOR_CLASS"
INVALID_BASE_MODEL = "INVALID_BASE_MODEL"
WEIGHT_SAVE_FAILURE = "WEIGHT_SAVE_FAILURE"

#: Default pretrained model used when the operator specifies none (Req 2.7).
DEFAULT_BASE_MODEL = "yolo26s.pt"


class TeacherTrainingError(Exception):
    """Raised when a teacher training request is rejected (Req 2.4, 2.6)."""


@dataclass
class TeacherTrainResult:
    """Outcome of a teacher training attempt."""

    target_class: int
    weights_path: str | None = None
    training_datasets: list[str] = field(default_factory=list)
    base_model: str = DEFAULT_BASE_MODEL
    succeeded: bool = False
    diagnostics: list[Diagnostic] = field(default_factory=list)


class TeacherTrainer:
    """Trains a Teacher_Model for a target class using only class-containing data (Req 2)."""

    def __init__(self, backend: TeacherBackend) -> None:
        self.backend = backend

    def datasets_containing(
        self, target_class: int, coverages: list[DatasetCoverage]
    ) -> list[str]:
        """Return dataset ids whose ground truth contains >= 1 instance of ``target_class`` (Req 2.1)."""
        result: list[str] = []
        for cov in coverages:
            for split_cov in cov.per_split.values():
                if target_class in split_cov.present:
                    result.append(cov.dataset_id)
                    break
        return result

    def train_teacher(
        self,
        target_class: int,
        coverages: list[DatasetCoverage],
        data_yaml: str,
        run_dir: str,
        seed: int,
        base_model: str | None = None,
        manifest=None,
    ) -> TeacherTrainResult:
        """Train a teacher for ``target_class`` (Req 2.1-2.8).

        Records training datasets before the first iteration (Req 2.2), rejects when no
        dataset contains the class (Req 2.4) or the base model is invalid (Req 2.6),
        defaults the base model when unspecified (Req 2.7), persists weights and records
        their location (Req 2.5), and on save failure records a diagnostic without marking
        success (Req 2.8).
        """
        result = TeacherTrainResult(target_class=target_class)

        # Req 2.1 / 2.4 - only class-containing datasets may train the teacher.
        training_datasets = self.datasets_containing(target_class, coverages)
        result.training_datasets = training_datasets
        if not training_datasets:
            result.diagnostics.append(
                Diagnostic(
                    code=NO_DATASET_FOR_CLASS,
                    message=f"no dataset contains ground truth for class {target_class}",
                    target=str(target_class),
                )
            )
            raise TeacherTrainingError(
                f"cannot train teacher for class {target_class}: no class-containing dataset"
            )

        # Req 2.3 / 2.6 / 2.7 - resolve and validate the base model.
        resolved_base = base_model if base_model else DEFAULT_BASE_MODEL
        result.base_model = resolved_base
        if base_model is not None and not self._is_valid_base_model(base_model):
            result.diagnostics.append(
                Diagnostic(
                    code=INVALID_BASE_MODEL,
                    message=f"base model {base_model!r} does not exist or is not a .pt file",
                    target=base_model,
                )
            )
            raise TeacherTrainingError(f"invalid base model: {base_model!r}")

        # Req 2.2 - record training datasets BEFORE the first iteration.
        if manifest is not None:
            manifest.training_datasets[target_class] = list(training_datasets)

        # Delegate the actual training to the backend (Req 2.5).
        try:
            weights_path = self.backend.train(
                data_yaml=data_yaml,
                base_model=resolved_base,
                seed=seed,
                run_dir=run_dir,
            )
        except Exception as exc:  # noqa: BLE001 - training/save failure (Req 2.8)
            result.diagnostics.append(
                Diagnostic(
                    code=WEIGHT_SAVE_FAILURE,
                    message=f"teacher training/persistence failed for class {target_class}: {exc}",
                    target=str(target_class),
                )
            )
            return result  # not marked succeeded (Req 2.8)

        result.weights_path = weights_path
        result.succeeded = True
        if manifest is not None:
            manifest.teacher_models[target_class] = weights_path  # Req 2.5
        return result

    @staticmethod
    def _is_valid_base_model(base_model: str) -> bool:
        """A base model is valid when it is an existing ``.pt`` file (mirrors train_yolov26)."""
        return base_model.lower().endswith(".pt") and os.path.isfile(base_model)
