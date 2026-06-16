"""Config-driven dataset preparation pipeline.

Usage:
    python -m prep_dataset          # prepare all datasets from config.yaml
    python -m prep_dataset plate    # prepare only the 'plate' dataset
"""

from prep_dataset.pipeline import run_pipeline  # noqa: F401
