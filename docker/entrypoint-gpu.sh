#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# GPU entrypoint: prepare datasets (idempotent), then run the command.
#
# Datasets are mounted from the host at /datasets.
# prep_dataset.py automatically skips any dataset that is already downloaded
# and extracted, so re-runs are cheap.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "==> Checking datasets..."
python prep_dataset.py || echo "[warn] dataset preparation had errors — continuing anyway"

echo "==> Running: $*"
exec "$@"