"""Zip extraction with structure detection and reorganization."""

from __future__ import annotations

import logging
import os
import subprocess
import zipfile
from pathlib import Path

from prep_dataset.structure import unwrap_if_needed

logger = logging.getLogger(__name__)


def extract_zip(zip_path: str, extract_path: str, expected_structure: list[str] | None = None) -> bool:
    """Extract a zip file and verify/reorganize the resulting structure.

    Handles different zip internal layouts:
    - Zips with a top-level folder matching the expected name
    - Zips with contents directly at root (no wrapper folder)
    - Zips with a differently-named wrapper folder

    Returns True on success, False on failure.
    """
    if not os.path.exists(zip_path):
        logger.error("[extract] zip file not found: %s", zip_path)
        return False

    extract_dir = Path(extract_path)
    parent_dir = extract_dir.parent

    # Determine what's inside the zip to choose extraction strategy
    try:
        top_level_dirs = _get_zip_top_level_dirs(zip_path)
    except (zipfile.BadZipFile, OSError) as e:
        logger.error("[extract] corrupt zip: %s — %s", zip_path, e)
        return False

    logger.info("[extract] %s -> %s (zip top-level: %s)", zip_path, extract_path, top_level_dirs)

    # Strategy: extract into parent and let structure detection sort it out
    if not _extract_to_dir(zip_path, str(parent_dir)):
        return False

    # If the zip extracted a single top-level folder with a different name,
    # rename it to our expected extract_path
    if not extract_dir.exists():
        # Check if zip created a differently-named folder
        if len(top_level_dirs) == 1:
            actual_dir = parent_dir / top_level_dirs[0]
            if actual_dir.exists() and actual_dir.is_dir():
                logger.info(
                    "[extract] renaming %s -> %s", actual_dir, extract_dir
                )
                actual_dir.rename(extract_dir)
            else:
                # Contents extracted flat into parent — create target and move
                logger.warning(
                    "[extract] expected %s but zip extracted to unknown location",
                    extract_path,
                )
                return False
        elif len(top_level_dirs) == 0:
            # Flat zip: files went directly into parent_dir — move into subfolder
            logger.info(
                "[extract] flat zip detected, creating %s", extract_path
            )
            extract_dir.mkdir(parents=True, exist_ok=True)
            # This case is tricky and depends on actual zip contents;
            # for now we trust the extraction went into parent correctly
        # else: multiple top-level dirs extracted into parent — this is a copyparty
        #       style zip where contents go directly to the target folder.
        #       We need to create the extract dir and populate it.

    # Handle copyparty-style zips that extract flat into parent
    # If extract_dir still doesn't exist, the zip had its contents directly
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        # Re-extract directly into the target
        if not _extract_to_dir(zip_path, str(extract_dir)):
            return False

    # Post-extraction: unwrap nested structure if needed
    if expected_structure:
        unwrap_if_needed(extract_path, expected_structure)

    return True


def _extract_to_dir(zip_path: str, dest_dir: str) -> bool:
    """Extract zip contents into dest_dir. Tries system unzip first, falls back to Python."""
    # Try system unzip (faster, preserves permissions)
    try:
        subprocess.run(
            ["unzip", "-q", "-o", zip_path, "-d", dest_dir],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except FileNotFoundError:
        logger.info("[extract] 'unzip' not found, falling back to Python zipfile")
    except subprocess.CalledProcessError as e:
        logger.warning("[extract] unzip failed (%s), falling back to Python zipfile", e.stderr.strip())

    # Python fallback
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        return True
    except (zipfile.BadZipFile, OSError) as e:
        logger.error("[extract] Python extraction failed: %s", e)
        return False


def _get_zip_top_level_dirs(zip_path: str) -> list[str]:
    """Inspect a zip to find top-level directory names."""
    top_level = set()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            # Get the first path component
            parts = name.split("/")
            if len(parts) > 1 and parts[0]:
                top_level.add(parts[0])
            elif len(parts) == 1 and not name.endswith("/"):
                # File at root level — no top-level dir
                top_level.add(".")
    return sorted(top_level)
