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
        project = os.path.join(run_dir, "teachers")

        # Memory/resume settings handed down by training/train_pipeline.py through the
        # environment (the teacher is a full training run and needs the same limits as the
        # other stages). Unset variables keep train()'s defaults.
        env = os.environ
        kwargs: dict = {}
        for var, key, cast in (
            ("PL_TEACHER_BATCH", "batch_size", int),
            ("PL_TEACHER_WORKERS", "workers", int),
            ("PL_TEACHER_IMGSZ", "imgsz", int),
            ("PL_TEACHER_EPOCHS", "epochs", int),
            ("PL_TEACHER_FRACTION", "fraction", float),
        ):
            if env.get(var):
                kwargs[key] = cast(env[var])
        if env.get("PL_TEACHER_LR0"):
            # Teachers start from an already trained model: a low learning rate fine-tunes
            # it instead of partly undoing it. optimizer="auto" would ignore lr0.
            kwargs["lr0"] = float(env["PL_TEACHER_LR0"])
            kwargs["optimizer"] = env.get("PL_TEACHER_OPTIMIZER") or "SGD"
        if env.get("PL_TEACHER_CACHE"):
            kwargs["cache"] = False if env["PL_TEACHER_CACHE"] == "off" else env["PL_TEACHER_CACHE"]
        if env.get("PL_TEACHER_RESUME") == "1":
            # Stable location + per-class name (data_yaml is teacher_class_<id>.yaml) so a
            # restarted pipeline finds this teacher again: an unfinished one is resumed from
            # its last.pt, a finished one is reused.
            project = env.get("PL_TEACHER_PROJECT") or os.path.join(
                os.path.dirname(run_dir.rstrip("/")) or ".", "teachers"
            )
            name = os.path.splitext(os.path.basename(data_yaml))[0] + "_run"
            kwargs["resume"] = True

        results = _train(
            data=data_yaml,
            model=base_model or "yolo26s.pt",
            project=project,
            name=name,
            **kwargs,
        )
        return os.path.join(str(results.save_dir), "weights", "best.pt")

    def load(self, weights_path: str) -> UltralyticsLoadedModel:
        from ultralytics import YOLO

        return UltralyticsLoadedModel(YOLO(weights_path), conf=self._conf)
