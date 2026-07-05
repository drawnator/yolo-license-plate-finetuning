"""Image -> label path mirroring (pure; no filesystem access).

Implements the Ultralytics directory convention used throughout the project: a label path
is derived from an image path by replacing the **last** ``images`` path segment with
``labels`` and swapping the image file extension for ``.txt`` while preserving the filename
stem (Req 11.2). Paths are treated as POSIX-style (forward-slash separated), matching the
dataset layout under ``datasets/``.
"""

from __future__ import annotations

from pathlib import PurePosixPath

#: The Ultralytics path segment that marks the image directory.
IMAGES_SEGMENT = "images"
#: The Ultralytics path segment that marks the label directory.
LABELS_SEGMENT = "labels"
#: The extension every YOLO label file uses.
LABEL_EXTENSION = ".txt"


def _last_index(parts: list[str], segment: str) -> int:
    """Return the index of the last occurrence of ``segment`` in ``parts``, or ``-1``."""
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == segment:
            return i
    return -1


def image_to_label_path(image_path: str) -> str:
    """Mirror an image path to its label path (Req 11.2).

    Replaces the **last** ``images`` segment with ``labels``, swaps the extension for
    ``.txt``, and preserves the filename stem (including embedded dots, so
    ``1013_jpg.rf.abc.jpg`` becomes ``1013_jpg.rf.abc.txt``).

    Raises:
        ValueError: if ``image_path`` contains no ``images`` path segment.
    """
    parts = list(PurePosixPath(image_path).parts)
    idx = _last_index(parts, IMAGES_SEGMENT)
    if idx == -1:
        raise ValueError(f"no {IMAGES_SEGMENT!r} path segment in image path: {image_path!r}")
    parts[idx] = LABELS_SEGMENT
    return str(PurePosixPath(*parts).with_suffix(LABEL_EXTENSION))
