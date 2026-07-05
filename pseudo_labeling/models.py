"""Core data models for the pseudo-labeling pipeline.

These are plain, mostly-frozen dataclasses over normalized YOLO data so that the pure
core logic (clipping, IoU dedup, formatting, thresholding) can be tested exhaustively
with property-based tests without any GPU or real model.

All bounding-box coordinates are normalized YOLO values (``x_center, y_center, width,
height``) in the inclusive range ``[0.0, 1.0]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class UnifiedClass(IntEnum):
    """The unified four-class label space defined in ``data.yaml``."""

    PLATE = 0
    FACE = 1
    CAR = 2
    MOTORCYCLE = 3


@dataclass(frozen=True)
class BBox:
    """Normalized YOLO bounding box; invariants enforced by :meth:`clip`."""

    x_center: float
    y_center: float
    width: float
    height: float

    def clip(self) -> "BBox":
        """Clip every coordinate into ``[0, 1]`` (Req 3.4, 11.3)."""

        def c(v: float) -> float:
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return v

        return BBox(c(self.x_center), c(self.y_center), c(self.width), c(self.height))

    def is_degenerate(self) -> bool:
        """True if width or height is ``0.0`` (Req 3.5)."""
        return self.width == 0.0 or self.height == 0.0


@dataclass(frozen=True)
class CandidateDetection:
    """A raw model detection (Req 3.1)."""

    class_id: int
    box: BBox
    confidence: float  # in [0.0, 1.0]


@dataclass(frozen=True)
class PseudoLabel:
    """A :class:`CandidateDetection` accepted for merging (post-threshold, post-resolution)."""

    class_id: int
    box: BBox
    confidence: float
    image_path: str


@dataclass(frozen=True)
class Diagnostic:
    """A structured diagnostic emitted by any component."""

    code: str  # e.g. "UNREADABLE_LABEL", "UNMAPPED_CLASS", "WRITE_FAILURE"
    message: str
    target: str  # dataset id, label path, or image path
