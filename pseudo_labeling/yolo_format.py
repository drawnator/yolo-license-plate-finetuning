"""YOLO label-line parsing and formatting helpers (Req 11).

A YOLO label line has the form ``<class_id> <x_center> <y_center> <width> <height>``
with fields separated by a single space (0x20), the class id a non-negative integer in
the :class:`~pseudo_labeling.models.UnifiedClass` space, and each of the four coordinate
values a decimal number formatted with exactly 6 digits after the decimal point.

These are pure functions over plain data so the formatting/parsing logic can be exercised
exhaustively with property-based tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import BBox, UnifiedClass

# Unified class id space {0, 1, 2, 3}. Derived from UnifiedClass to avoid coupling to a
# separate unified.py module (which may be authored in parallel).
UNIFIED_IDS: frozenset[int] = frozenset(int(c) for c in UnifiedClass)


@dataclass(frozen=True)
class LabelLine:
    """A single parsed YOLO label line.

    ``raw`` preserves the original line text so ground-truth lines can be round-tripped
    back to disk byte-for-byte (Req 5.1).
    """

    class_id: int
    box: BBox
    raw: str | None = None


def format_line(class_id: int, box: BBox) -> str:
    """Format a class id and box as a YOLO label line (Req 11.1, 11.3, 11.4).

    The box is clamped to ``[0, 1]`` before formatting (Req 11.3, 11.4) and each of the
    four coordinates is rendered with exactly 6 digits after the decimal point. Fields are
    separated by a single space (0x20).

    Raises:
        ValueError: if ``class_id`` is not in the unified class space ``{0, 1, 2, 3}``
            (Req 11.5). The caller decides to skip the offending line.
    """
    if class_id not in UNIFIED_IDS:
        raise ValueError(
            f"class id {class_id!r} is not in the unified class space {sorted(UNIFIED_IDS)}"
        )

    clipped = box.clip()
    return (
        f"{int(class_id)} "
        f"{clipped.x_center:.6f} "
        f"{clipped.y_center:.6f} "
        f"{clipped.width:.6f} "
        f"{clipped.height:.6f}"
    )


def parse_line(text: str) -> LabelLine:
    """Parse a YOLO label line into a :class:`LabelLine` (Req 1.6).

    The original ``text`` is preserved on the returned :class:`LabelLine` as ``raw`` so
    ground-truth lines can be written back unchanged.

    Raises:
        ValueError: if the class id field is missing or malformed, or if the geometry
            fields are missing or malformed (Req 1.6).
    """
    fields = text.split()
    if len(fields) != 5:
        raise ValueError(
            f"malformed YOLO label line: expected 5 space-separated fields, got "
            f"{len(fields)}: {text!r}"
        )

    class_id_field, *coord_fields = fields

    try:
        class_id = int(class_id_field)
    except ValueError as exc:
        raise ValueError(
            f"malformed class id {class_id_field!r} in YOLO label line: {text!r}"
        ) from exc

    try:
        x_center, y_center, width, height = (float(v) for v in coord_fields)
    except ValueError as exc:
        raise ValueError(
            f"malformed geometry fields {coord_fields!r} in YOLO label line: {text!r}"
        ) from exc

    return LabelLine(
        class_id=class_id,
        box=BBox(x_center, y_center, width, height),
        raw=text,
    )
