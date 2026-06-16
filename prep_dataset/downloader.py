"""Robust download logic with fallback URLs and skip-if-exists support."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def download_with_fallback(urls: List[str], dest_path: str) -> bool:
    """Try each URL in order until one succeeds. Returns True on success."""
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    for i, url in enumerate(urls):
        logger.info(
            "[download] trying URL %d/%d: %s -> %s", i + 1, len(urls), url, dest_path
        )
        try:
            result = subprocess.run(
                ["curl", "-L", "--fail", "-o", dest_path, url],
                check=True,
                capture_output=True,
                text=True,
            )
            # Verify file was actually created and isn't empty
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                logger.info("[download] success from URL %d: %s", i + 1, url)
                return True
            else:
                logger.warning(
                    "[download] URL %d produced empty file, trying next", i + 1
                )
                _remove_if_exists(dest_path)
        except subprocess.CalledProcessError as e:
            logger.warning(
                "[download] URL %d failed: %s", i + 1, e.stderr.strip() or str(e)
            )
            _remove_if_exists(dest_path)

    logger.error("[download] all %d URLs failed for %s", len(urls), dest_path)
    return False


def _remove_if_exists(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
