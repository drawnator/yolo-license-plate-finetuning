"""TeacherBackend interface and the real Ultralytics implementation.

Teacher model access (training + inference) is abstracted behind the :class:`TeacherBackend`
and :class:`LoadedModel` protocols so the rest of the pipeline depends only on a small,
stable seam. This makes it straightforward to add a test double later (see the deferred
testing notes in ``pseudo_labeling/DEFERRED_WORK.md``) without touching pipeline code.

Only the real :class:`UltralyticsBackend` ships today: it wraps ``ultralytics.YOLO`` for
inference and delegates training to ``training/train_yolov26.py::train`` so augmentation and
MLflow behavior stay consistent with the rest of the project.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import CandidateDetection


@runtime_checkable
class LoadedModel(Protocol):
    """A loaded teacher model capable of running inference on a single image."""

    def infer(self, image_path: str) -> list[CandidateDetection]:
        """Return normalized detections for an image (Req 3.1).

        Raises on inference failure — callers (the generator) turn a raised exception into
        a diagnostic and continue (Req 3.7).
        """
        ...


@runtime_checkable
class TeacherBackend(Protocol):
    """Abstraction over teacher training and model loading."""

    def train(self, data_yaml: str, base_model: str | None, seed: int, run_dir: str) -> str:
        """Train a teacher and return the persisted weights path (Req 2.5)."""
        ...

    def load(self, weights_path: str) -> LoadedModel:
        """Load a persisted teacher model for inference."""
        ...


class UltralyticsLoadedModel:
    """A loaded Ultralytics model that returns normalized detections (Req 3.1)."""

    def __init__(self, model, conf: float = 0.001) -> None:
        self._model = model
        self._conf = conf

    def infer(self, image_path: str) -> list[CandidateDetection]:
        from .models import BBox

        results = self._model.predict(source=image_path, conf=self._conf, verbose=False)
        detections: list[CandidateDetection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for (xc, yc, w, h), c, p in zip(
                boxes.xywhn.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
            ):
                detections.append(
                    CandidateDetection(
                        class_id=int(c),
                        box=BBox(float(xc), float(yc), float(w), float(h)),
                        confidence=float(p),
                    )
                )
        return detections


class UltralyticsBackend:
    """Real teacher backend wrapping ``ultralytics.YOLO`` (Req 2.5, 2.7, 3.1).

    Training is delegated to ``training/train_yolov26.py::train(data=...)``; inference wraps
    ``YOLO.predict`` and returns normalized :class:`CandidateDetection` values.
    """

    def __init__(self, conf: float = 0.001) -> None:
        self._conf = conf

    def train(self, data_yaml: str, base_model: str | None, seed: int, run_dir: str) -> str:
        import os

        from training.train_yolov26 import train as _train

        name = f"teacher_{os.path.basename(run_dir) or 'run'}"
        results = _train(
            data=data_yaml,
            model=base_model or "yolo26s.pt",
            project=os.path.join(run_dir, "teachers"),
            name=name,
        )
        return os.path.join(str(results.save_dir), "weights", "best.pt")

    def load(self, weights_path: str) -> UltralyticsLoadedModel:
        from ultralytics import YOLO

        return UltralyticsLoadedModel(YOLO(weights_path), conf=self._conf)
