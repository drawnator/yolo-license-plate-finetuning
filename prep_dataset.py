"""Legacy entry point — delegates to the prep_dataset package.

Usage:
    python prep_dataset.py              # prepare all datasets
    python prep_dataset.py plate ALPR   # prepare only named datasets

For the full module interface:
    python -m prep_dataset [dataset_name ...]
"""

import sys
from prep_dataset.pipeline import run_pipeline


def main():
    dataset_names = sys.argv[1:] if len(sys.argv) > 1 else None
    run_pipeline(dataset_names=dataset_names)


if __name__ == "__main__":
    main()
