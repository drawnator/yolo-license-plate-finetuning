"""Pseudo_Label_Generator and confidence filtering (Req 3, 4).

The generator runs a loaded teacher model over target images and turns raw detections
into clean :class:`~pseudo_labeling.models.CandidateDetection` values: only the requested
(absent) target classes are kept, boxes are clipped into ``[0, 1]``, and degenerate
zero-area boxes are dropped (Req 3). Confidence filtering (Req 4) is provided by the pure
:func:`resolve_threshold` / :func:`accepts` helpers.

Errors are handled defensively (Req 3.6, 3.7): an unreadable image is skipped and an
inference failure on a readable image yields zero detections; both record a
:class:`~pseudo_labeling.models.Diagnostic` and let processing continue.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .backends import LoadedModel
from .config import ThresholdConfig
from .models import CandidateDetection, Diagnostic

#: Diagnostic code for an image that could not be read (Req 3.6).
UNREADABLE_IMAGE = "UNREADABLE_IMAGE"
#: Diagnostic code for a teacher inference failure on a readable image (Req 3.7).
INFERENCE_FAILURE = "INFERENCE_FAILURE"


class NoThresholdConfigured(Exception):
    """Raised when neither a per-class nor a default Confidence_Threshold exists (Req 4.5)."""


def resolve_threshold(class_id: int, cfg: ThresholdConfig) -> float:
    """Resolve the Confidence_Threshold for ``class_id`` (Req 4.1, 4.3).

    Returns the per-class threshold when configured, otherwise the configured default.

    Raises:
        NoThresholdConfigured: when neither a per-class value for ``class_id`` nor a
            default is configured (Req 4.5).
    """
    if class_id in cfg.per_class:
        return cfg.per_class[class_id]
    if cfg.default is not None:
        return cfg.default
    raise NoThresholdConfigured(
        f"no Confidence_Threshold configured for class {class_id} and no default set"
    )


def accepts(candidate: CandidateDetection, threshold: float) -> bool:
    """Return ``True`` when the candidate's confidence meets the threshold (Req 4.1).

    The boundary is inclusive: a candidate is accepted iff ``confidence >= threshold``.
    """
    return candidate.confidence >= threshold


@dataclass
class PseudoLabelGenerator:
    """Runs a teacher model over images to produce Candidate_Detections (Req 3).

    Diagnostics for skipped/failed images accumulate on :attr:`diagnostics` so the caller
    (the orchestrator) can fold them into the Audit_Report.
    """

    diagnostics: list[Diagnostic] = field(default_factory=list)

    def generate_for_image(
        self,
        model: LoadedModel,
        image_path: str,
        target_classes: frozenset[int],
    ) -> list[CandidateDetection]:
        """Produce clean Candidate_Detections for one image (Req 3.1-3.7).

        Keeps only detections whose class is in ``target_classes`` (Req 3.2), clips each
        box into ``[0, 1]`` (Req 3.4), and drops degenerate boxes (Req 3.5). An unreadable
        image is skipped with a diagnostic (Req 3.6); an inference failure yields zero
        detections with a diagnostic and processing continues (Req 3.7).
        """
        if not self._is_readable(image_path):
            self.diagnostics.append(
                Diagnostic(
                    code=UNREADABLE_IMAGE,
                    message=f"skipping unreadable image {image_path!r}",
                    target=image_path,
                )
            )
            return []

        try:
            raw = model.infer(image_path)
        except Exception as exc:  # noqa: BLE001 - any inference failure is contained (Req 3.7)
            self.diagnostics.append(
                Diagnostic(
                    code=INFERENCE_FAILURE,
                    message=f"inference failed on image {image_path!r}: {exc}",
                    target=image_path,
                )
            )
            return []

        results: list[CandidateDetection] = []
        for det in raw:
            if det.class_id not in target_classes:
                continue  # Req 3.2 - only absent target classes.
            clipped = det.box.clip()  # Req 3.4
            if clipped.is_degenerate():
                continue  # Req 3.5 - drop zero-area boxes.
            results.append(
                CandidateDetection(
                    class_id=det.class_id,
                    box=clipped,
                    confidence=det.confidence,
                )
            )
        return results

    @staticmethod
    def _is_readable(image_path: str) -> bool:
        """Best-effort readability check used to distinguish Req 3.6 from Req 3.7."""
        return os.path.isfile(image_path) and os.access(image_path, os.R_OK)
