#!/usr/bin/env bash
# Full three-stage training, sized for an RTX 4060 Ti (8 GB VRAM) and a 48 GB WSL VM.
# Settings come from the memory tests of 2026-11-15 (see runs/logs/memtest-*.log).
#
#   Start / continue after a crash:  ./run_full.sh
#   Follow the log:                  tail -f runs/logs/yolo-train.log
#   Stop:                            docker stop yolo-train
#   After a crash:                   docker inspect yolo-train --format '{{.State.OOMKilled}}'
#                                    (true = RAM limit hit; false + "CUDA error: out of memory"
#                                    in the log = VRAM)
#
# Re-running is safe: --resume continues an interrupted stage from its last.pt
# (saved every epoch under angelicam/full/) and reuses stages that already finished.
set -euo pipefail
cd "$(dirname "$0")"

# Overridable for smoke tests, e.g.:
#   TRAIN_NAME=memtest-e2e TRAIN_PROJECT=angelicam/memtest/e2e LABEL_SETS_DIR=runs/memtest_label_sets \
#   MLFLOW_TRACKING_URI= EXTRA_ARGS="--fraction 0.02 --pretrain-epochs 1 --final-epochs 1" ./run_full.sh
NAME=${TRAIN_NAME:-yolo-train}      # not $NAME: WSL sets that to the hostname
PROJECT=${TRAIN_PROJECT:-angelicam/full}
LABEL_SETS_DIR=${LABEL_SETS_DIR:-datasets/label_sets}   # where pseudo-labels are stored
EXTRA_ARGS=${EXTRA_ARGS:-}
BATCH=${BATCH:-4}               # yolo26l stages (pretrain + pseudo-label teachers)
FINAL_BATCH=${FINAL_BATCH:-12}   # yolo26s final stage; use 8 if the desktop needs more VRAM
TEACHER_EPOCHS=${TEACHER_EPOCHS:-30}  # pseudo-label teachers start from the pretrain; 100 is overkill

mkdir -p datasets "$LABEL_SETS_DIR" runs/logs angelicam

if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ]; then
  echo "$NAME is already running. Stop it first: docker stop $NAME" >&2
  exit 1
fi

docker build -t yolo-train:latest -f docker/Dockerfile .
docker rm "$NAME" >/dev/null 2>&1 || true

docker run -d --gpus all --memory=44g --memory-swap=60g --shm-size=8g \
  --name "$NAME" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI-https://mlflow.guilherme.zip}" \
  -e MLFLOW_TRACKING_USERNAME="${MLFLOW_TRACKING_USERNAME:-admin}" \
  -e MLFLOW_TRACKING_PASSWORD="${MLFLOW_TRACKING_PASSWORD:-}" \
  -e MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true \
  -e GIT_COMMIT_HASH="$(git rev-parse HEAD 2>/dev/null || echo unknown)" \
  -v "$PWD/datasets:/datasets" \
  -v "$PWD/runs:/runs" \
  -v "$PWD/angelicam:/angelicam" \
  -v "$PWD/$LABEL_SETS_DIR:/datasets/label_sets" \
  yolo-train:latest \
  bash -c "set -o pipefail; python -m training.train_pipeline \
    --synthetic 50000 --pretrain --skip-export --mix-synthetic \
    --project $PROJECT \
    --batch-size $BATCH --final-batch-size $FINAL_BATCH --teacher-epochs $TEACHER_EPOCHS \
    --imgsz 640 --workers 8 --cache off \
    --resume $EXTRA_ARGS \
    2>&1 | tee -a /runs/logs/$NAME.log"

echo "Started $NAME. Log: runs/logs/$NAME.log"
