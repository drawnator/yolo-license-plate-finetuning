"""Main pipeline: orchestrates config loading, downloading, extraction, and conversion."""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional

from prep_dataset.config import DatasetConfig, load_config
from prep_dataset.converters import get_converter
from prep_dataset.downloader import download_with_fallback
from prep_dataset.extractor import extract_zip
from prep_dataset.structure import dataset_already_valid

logger = logging.getLogger(__name__)


def run_pipeline(
    config_path: str = "config.yaml",
    dataset_names: Optional[List[str]] = None,
) -> dict:
    """Run the full dataset preparation pipeline.

    Args:
        config_path: Path to the YAML config file.
        dataset_names: If provided, only process these dataset names.
                       If None, process all datasets in the config.

    Returns:
        A summary dict: {"succeeded": [...], "failed": [...], "skipped": [...]}
    """
    _setup_logging()

    datasets = load_config(config_path)

    if dataset_names:
        name_set = set(dataset_names)
        datasets = [d for d in datasets if d.name in name_set]
        if not datasets:
            logger.error("No matching datasets found for: %s", dataset_names)
            return {"succeeded": [], "failed": list(dataset_names), "skipped": []}

    os.makedirs("datasets", exist_ok=True)

    succeeded = []
    failed = []
    skipped = []

    for ds in datasets:
        logger.info("=" * 60)
        logger.info("[pipeline] processing dataset: %s", ds.name)
        logger.info("=" * 60)

        try:
            result = _process_dataset(ds)
            if result == "skipped":
                skipped.append(ds.name)
            elif result == "success":
                succeeded.append(ds.name)
            else:
                failed.append(ds.name)
        except Exception as e:
            logger.error("[pipeline] unexpected error for %s: %s", ds.name, e, exc_info=True)
            failed.append(ds.name)

    # Summary
    logger.info("=" * 60)
    logger.info("[pipeline] DONE — succeeded=%s skipped=%s failed=%s",
                succeeded, skipped, failed)
    logger.info("=" * 60)

    if failed:
        logger.warning(
            "[pipeline] %d dataset(s) failed but pipeline continues with the rest: %s",
            len(failed), failed,
        )

    return {"succeeded": succeeded, "failed": failed, "skipped": skipped}


def _process_dataset(ds: DatasetConfig) -> str:
    """Process a single dataset. Returns 'skipped', 'success', or 'failed'."""

    # Step 1: Check if valid dataset already exists (flexible depth matching)
    if dataset_already_valid(ds.extract_path, ds.expected_structure):
        logger.info("[pipeline] %s: valid dataset found at %s, skipping", ds.name, ds.extract_path)
        return "skipped"

    # Step 2: Download (with fallbacks) — skip if zip already exists
    if os.path.exists(ds.zip_path) and os.path.getsize(ds.zip_path) > 0:
        logger.info("[pipeline] %s: zip already exists at %s, skipping download", ds.name, ds.zip_path)
    else:
        if not ds.urls:
            logger.error("[pipeline] %s: no download URLs configured", ds.name)
            return "failed"

        success = download_with_fallback(ds.urls, ds.zip_path)
        if not success:
            return "failed"

    # Step 3: Extract
    if not extract_zip(ds.zip_path, ds.extract_path, ds.expected_structure):
        return "failed"

    # Step 4: Run converter if specified
    converter = get_converter(ds.converter)
    if converter:
        logger.info("[pipeline] %s: running converter '%s'", ds.name, ds.converter)
        converter(ds.extract_path)

    return "success"


def _setup_logging():
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
