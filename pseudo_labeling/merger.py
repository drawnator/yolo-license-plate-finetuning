"""Label_Merger with pluggable output targets (Req 5, 7, 11, 13).

The merge algorithm is identical regardless of destination; only *where* the merged label
file is written (and where its seed content comes from) differs. That destination is
abstracted behind :class:`LabelTarget`:

- :class:`InPlaceTarget` writes into the dataset's own ``labels`` directory and seeds from
  that same file (Req 5, 7, 11).
- :class:`LabelSetTarget` writes into a separate label-set tree, seeding each destination
  byte-for-byte from the ORIGINAL labels so the originals are never opened for writing
  (Req 13).

Ground truth is preserved byte-for-byte and pseudo-labels are appended after it; a
pseudo-label is dropped when its class already exists as ground truth for the image
(Req 5.3) or when it duplicates an existing same-class box by IoU (Req 5.4). Writes are
atomic (temp file + rename) so a failure never leaves a partial file.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from .models import BBox, Diagnostic, PseudoLabel
from .paths import image_to_label_path
from .yolo_format import LabelLine, format_line, parse_line

#: Default duplicate-overlap IoU threshold when unconfigured (Req 5.4).
DEFAULT_DUP_IOU = 0.5

#: Diagnostic code for a label-file write/append failure (Req 5.6, 7.7, 11.6).
WRITE_FAILURE = "WRITE_FAILURE"
#: Diagnostic code for a label-set seed (copy of original) failure (Req 13.9).
LABEL_SET_SEED_FAILURE = "LABEL_SET_SEED_FAILURE"
#: Diagnostic code for a pseudo-label whose class id is not unified (Req 11.5).
NON_UNIFIED_CLASS = "NON_UNIFIED_CLASS"


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two normalized YOLO boxes.

    Returns ``0.0`` when either box is degenerate or the boxes do not overlap.
    """
    if a.is_degenerate() or b.is_degenerate():
        return 0.0

    ax1, ax2 = a.x_center - a.width / 2, a.x_center + a.width / 2
    ay1, ay2 = a.y_center - a.height / 2, a.y_center + a.height / 2
    bx1, bx2 = b.x_center - b.width / 2, b.x_center + b.width / 2
    by1, by2 = b.y_center - b.height / 2, b.y_center + b.height / 2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0

    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def is_duplicate(
    candidate: PseudoLabel,
    existing: list[LabelLine],
    dup_threshold: float = DEFAULT_DUP_IOU,
) -> bool:
    """Return ``True`` when ``candidate`` overlaps a same-class existing label (Req 5.4).

    A candidate is a duplicate when it shares a class with an existing label and their IoU
    is greater than or equal to ``dup_threshold``.
    """
    for line in existing:
        if line.class_id == candidate.class_id and iou(candidate.box, line.box) >= dup_threshold:
            return True
    return False


@runtime_checkable
class LabelTarget(Protocol):
    """Resolves the destination and seed for a merged label file (Req 13.3, 13.7)."""

    def resolve_target_label_path(self, image_path: str, dataset_id: str, split: str) -> str:
        """Return the destination label path for ``image_path``."""
        ...

    def seed_original_label_path(
        self, image_path: str, dataset_id: str, split: str
    ) -> str | None:
        """Return the path whose bytes seed the destination, or ``None`` if not applicable."""
        ...


class InPlaceTarget:
    """Write into the dataset's own labels directory; the file self-seeds (Req 5, 7, 11)."""

    def resolve_target_label_path(self, image_path: str, dataset_id: str, split: str) -> str:
        return image_to_label_path(image_path)

    def seed_original_label_path(
        self, image_path: str, dataset_id: str, split: str
    ) -> str | None:
        # In place, the destination *is* the original: it self-seeds.
        return image_to_label_path(image_path)


class LabelSetTarget:
    """Write into a separate label-set tree, seeding from the ORIGINAL labels (Req 13).

    The destination mirrors the image path (``images`` -> ``labels``, ``.txt`` extension)
    but rooted under ``label_set_root``, preserving the dataset id and per-split layout so
    the label set is YOLO-consumable. Originals are never opened for writing.
    """

    def __init__(self, label_set_root: str, datasets_root: str = "datasets") -> None:
        self.label_set_root = label_set_root
        self.datasets_root = datasets_root

    def resolve_target_label_path(self, image_path: str, dataset_id: str, split: str) -> str:
        original_label = image_to_label_path(image_path)
        rel = self._relative_to_root(original_label)
        return str(PurePosixPath(self.label_set_root, rel))

    def seed_original_label_path(
        self, image_path: str, dataset_id: str, split: str
    ) -> str | None:
        return image_to_label_path(image_path)

    def _relative_to_root(self, label_path: str) -> str:
        """Return ``label_path`` relative to ``datasets_root`` (keeping dataset id + split)."""
        parts = PurePosixPath(label_path).parts
        root_parts = PurePosixPath(self.datasets_root).parts
        if parts[: len(root_parts)] == root_parts:
            return str(PurePosixPath(*parts[len(root_parts) :]))
        # Fall back to stripping a leading "datasets" segment if present.
        if parts and parts[0] == "datasets":
            return str(PurePosixPath(*parts[1:]))
        return str(PurePosixPath(*parts))


@dataclass
class MergeResult:
    """Outcome of merging one image's labels."""

    image_path: str
    target_path: str
    written: bool = False
    created: bool = False
    pseudo_added: int = 0
    skipped_existing_class: int = 0
    skipped_duplicate: int = 0
    skipped_non_unified: int = 0
    diagnostics: list[Diagnostic] = field(default_factory=list)


class LabelMerger:
    """Merge accepted pseudo-labels into label files via an injected :class:`LabelTarget`."""

    def __init__(self, target: LabelTarget) -> None:
        self.target = target

    def merge_image(
        self,
        image_path: str,
        dataset_id: str,
        split: str,
        accepted: list[PseudoLabel],
        dup_threshold: float = DEFAULT_DUP_IOU,
    ) -> MergeResult:
        """Merge ``accepted`` pseudo-labels for one image (Req 5, 7, 11, 13).

        Ground-truth lines from the seed are preserved byte-for-byte and appended to;
        pseudo-labels of a class already present as ground truth are skipped (Req 5.3),
        as are IoU duplicates (Req 5.4) and non-unified class ids (Req 11.5).
        """
        target_path = self.target.resolve_target_label_path(image_path, dataset_id, split)
        seed_path = self.target.seed_original_label_path(image_path, dataset_id, split)
        result = MergeResult(image_path=image_path, target_path=target_path)

        # --- Read seed (original ground truth), preserving exact bytes. ---
        seed_text: str | None = None
        if seed_path is not None and os.path.isfile(seed_path):
            try:
                with open(seed_path, "r", encoding="utf-8") as handle:
                    seed_text = handle.read()
            except OSError as exc:
                result.diagnostics.append(
                    Diagnostic(
                        code=LABEL_SET_SEED_FAILURE,
                        message=f"could not read original label {seed_path!r}: {exc}",
                        target=seed_path,
                    )
                )
                return result  # leave originals untouched, exclude image (Req 13.9)

        existing_lines = self._parse_existing(seed_text)
        existing_classes = {line.class_id for line in existing_lines}

        # --- Filter accepted pseudo-labels against existing labels. ---
        new_lines: list[str] = []
        kept: list[LabelLine] = list(existing_lines)
        for pl in accepted:
            if pl.class_id in existing_classes:
                result.skipped_existing_class += 1  # Req 5.3
                continue
            if is_duplicate(pl, kept, dup_threshold):
                result.skipped_duplicate += 1  # Req 5.4
                continue
            try:
                line = format_line(pl.class_id, pl.box)  # Req 11.1/11.3/11.5
            except ValueError:
                result.skipped_non_unified += 1  # Req 11.5 - skip, keep going
                result.diagnostics.append(
                    Diagnostic(
                        code=NON_UNIFIED_CLASS,
                        message=f"skipping non-unified class id {pl.class_id} for {image_path!r}",
                        target=image_path,
                    )
                )
                continue
            new_lines.append(line)
            kept.append(parse_line(line))

        result.pseudo_added = len(new_lines)

        # --- Decide whether to write. ---
        is_label_set = seed_path != target_path
        original_exists = seed_text is not None

        if not new_lines:
            # No pseudo-labels to add.
            if is_label_set and original_exists:
                # Still copy the original into the label set (complete copy, Req 13.4).
                self._write(target_path, seed_text, result, created=not os.path.exists(target_path))
            # Otherwise leave things unchanged (Req 7.1/7.2) / create nothing (Req 7.3).
            return result

        # We have pseudo-labels to write.
        created = not os.path.exists(target_path) if not is_label_set else (not original_exists and not os.path.exists(target_path))
        content = self._compose(seed_text, new_lines)
        self._write(target_path, content, result, created=created or (original_exists is False))
        return result

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_existing(seed_text: str | None) -> list[LabelLine]:
        """Parse seed text into LabelLines, ignoring blank lines. Malformed -> treated empty."""
        if not seed_text:
            return []
        lines: list[LabelLine] = []
        for raw in seed_text.splitlines():
            if not raw.strip():
                continue
            try:
                lines.append(parse_line(raw))
            except ValueError:
                # A malformed ground-truth line is preserved as-is (no box for dedup).
                lines.append(LabelLine(class_id=-1, box=BBox(0, 0, 0, 0), raw=raw))
        return lines

    @staticmethod
    def _compose(seed_text: str | None, new_lines: list[str]) -> str:
        """Compose final file content: original bytes preserved, new lines appended."""
        appended = "\n".join(new_lines) + "\n"
        if not seed_text:
            return appended
        # Preserve the original bytes exactly; ensure a separating newline.
        if seed_text.endswith("\n"):
            return seed_text + appended
        return seed_text + "\n" + appended

    @staticmethod
    def _write(path: str, content: str, result: MergeResult, created: bool) -> None:
        """Atomically write ``content`` to ``path`` (temp file + rename)."""
        try:
            parent = os.path.dirname(path) or "."
            os.makedirs(parent, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(content)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            result.written = True
            result.created = created
        except OSError as exc:
            result.diagnostics.append(
                Diagnostic(
                    code=WRITE_FAILURE,
                    message=f"failed to write label file {path!r}: {exc}",
                    target=path,
                )
            )
