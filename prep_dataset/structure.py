"""Flexible structure detection and validation for extracted datasets.

Key features:
- Detect expected structure at any nesting depth (handles wrapper folders,
  renamed roots, extra levels after extraction).
- Match on structure pattern rather than exact root folder name.
- Reorganize files if the expected structure is found deeper than expected.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Known structure patterns to look for when validating a dataset exists
YOLO_SPLIT_MARKERS = {"train", "valid", "test"}
YOLO_DATA_MARKERS = {"images", "labels"}


def dataset_already_valid(extract_path: str, expected_structure: List[str]) -> bool:
    """Check if the extract_path contains a valid dataset.

    Uses flexible depth matching: looks for expected_structure markers at any
    depth within the extract_path tree. If no expected_structure is specified,
    falls back to common patterns (train/valid/test or images/labels).

    Returns True if the dataset looks usable and shouldn't be re-downloaded.
    """
    root = Path(extract_path)
    if not root.exists() or not root.is_dir():
        return False

    # An empty directory is not valid
    if not any(root.iterdir()):
        return False

    markers = set(expected_structure) if expected_structure else None

    if markers:
        return _find_markers_in_tree(root, markers)

    # No explicit markers: try common patterns
    # 1. YOLO-style splits (train/valid/test)
    if _find_markers_in_tree(root, YOLO_SPLIT_MARKERS, min_matches=2):
        return True
    # 2. YOLO data dirs (images/labels)
    if _find_markers_in_tree(root, YOLO_DATA_MARKERS, min_matches=2):
        return True
    # 3. Has actual image files somewhere (broad fallback)
    if _has_image_files(root):
        return True

    return False


def _find_markers_in_tree(
    root: Path, markers: set, min_matches: int = None
) -> bool:
    """Walk the directory tree and check if marker directory names exist.

    Args:
        root: Root directory to search.
        markers: Set of directory names to look for.
        min_matches: Minimum number of markers that must be found.
                     Defaults to all markers.
    """
    if min_matches is None:
        min_matches = len(markers)

    found = set()
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d in markers:
                found.add(d)
                if len(found) >= min_matches:
                    return True
    return len(found) >= min_matches


def _has_image_files(root: Path) -> bool:
    """Check if any image files exist anywhere in the tree."""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for _, _, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() in image_exts:
                return True
    return False


def find_structure_root(extract_path: str, expected_structure: List[str]) -> Optional[Path]:
    """Find the deepest directory that contains all expected structure markers.

    Useful for detecting when extraction added an extra wrapper folder.
    Returns the path to the directory that directly contains the markers,
    or None if not found.
    """
    root = Path(extract_path)
    if not root.exists():
        return None

    markers = set(expected_structure)
    if not markers:
        return root

    # BFS from root downward
    for dirpath, dirnames, _ in os.walk(root):
        dir_names_set = set(dirnames)
        if markers.issubset(dir_names_set):
            return Path(dirpath)

    return None


def unwrap_if_needed(extract_path: str, expected_structure: List[str]) -> None:
    """If the dataset is nested one level deeper than expected, unwrap it.

    Common scenario: zip extracts to extract_path/SomeFolder/train/...
    but we want extract_path/train/...
    """
    root = Path(extract_path)
    if not root.exists():
        return

    markers = set(expected_structure) if expected_structure else None
    if not markers:
        return

    # Check if markers are directly in root
    direct_dirs = {d.name for d in root.iterdir() if d.is_dir()}
    if markers.issubset(direct_dirs):
        return  # Already at correct level

    # Look one level down
    structure_root = find_structure_root(extract_path, expected_structure)
    if structure_root and structure_root != root:
        logger.info(
            "[structure] unwrapping: moving contents from %s to %s",
            structure_root,
            root,
        )
        _move_contents_up(structure_root, root)


def _move_contents_up(source: Path, target: Path) -> None:
    """Move all contents of source into target, handling conflicts."""
    for item in source.iterdir():
        dest = target / item.name
        if dest.exists():
            if dest.is_dir() and item.is_dir():
                # Merge directories
                _move_contents_up(item, dest)
                item.rmdir()
            else:
                # Overwrite file
                if dest.is_file():
                    dest.unlink()
                shutil.move(str(item), str(dest))
        else:
            shutil.move(str(item), str(dest))

    # Clean up empty source directories
    try:
        source.rmdir()
    except OSError:
        pass
