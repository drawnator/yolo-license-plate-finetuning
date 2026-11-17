# Training on bh02 (RTX 4060 Ti 8 GB, WSL2)

Findings from the memory tests of 2026-11-15 and the full runs started on 2026-11-16.
They explain the settings in `run_full.sh`, which is the launcher to use on this machine.

## Short version

```bash
./run_full.sh                       # start, or continue after a crash
tail -f runs/logs/yolo-train.log    # follow
docker stop yolo-train              # stop (re-running resumes from the last epoch)
```

| Setting | Value | Why |
|---|---|---|
| Batch, `yolo26l` stages (pretrain, pseudo-label teachers) | 4 | Batch 6 fills the card, batch 8 overflows it |
| Batch, `yolo26s` final stage | 12 | Batch 16 is at the limit; use `FINAL_BATCH=8` if the desktop needs VRAM |
| Image size | 640 | Unchanged from the project default |
| Workers | 8 | RAM is not the constraint, so no reason to lower it |
| Cache | off | RAM cache does not fit the full set and gave no speed-up |
| Container limits | `--memory=44g --memory-swap=60g --shm-size=8g` | Keeps a runaway process from taking down the WSL VM |

The limit on this machine is VRAM, not RAM. With these settings the full run uses about
8 GB of RAM and 5.3 to 5.9 GB of the card (measured in the first attempt).

## Machine

- 56 GB RAM, WSL2 capped at 48 GB by `.wslconfig`.
- RTX 4060 Ti, 8 GB VRAM. The Windows desktop holds about 1.9 GB of it, so training has
  roughly 6 GB.
- Docker Desktop with WSL integration, GPU through `--gpus all`.
- The project lives in the WSL filesystem (`~/projects/yolo-license-plate-finetuning`),
  not under `/mnt/c`.

## What uses the memory

**VRAM.** On WSL, going past the card's memory does not fail right away. The driver spills
into Windows shared GPU memory and training keeps running 4 to 20 times slower; only later
does it end in a CUDA out-of-memory error. The `GPU_mem` column in the Ultralytics log is
the number to watch: if it is above about 6G on this machine, the run is in overflow.

This is what killed the run of 2026-11-13: `yolo26l` at batch 16 with `cache=True` ended
in `CUDA_ERROR_OUT_OF_MEMORY`. It was not a RAM kill.

**RAM.** The image cache is the only large consumer. The training set is 60,549 images
(42,500 of them synthetic) plus 10,782 validation images.

- `cache=True` on the full training set needs about 64 GB. Ultralytics asks for 128.5 GB
  with its safety margin and silently declines to cache.
- The validation set (about 10 GB) does get cached, which adds 16 GB of RAM for no gain
  in speed.
- With the cache off, RAM stays between 8 and 14 GB at 8 workers.

## Tests

All at 640 px. "Torch" is the `GPU_mem` column; "card" is `nvidia-smi` and includes the
1.9 GB desktop share. Tests on 2% of the data are for memory only, so their epoch times
are not listed.

| Test | Model, batch | Data | Cache | Peak RAM | VRAM card / torch | Time per epoch | Result |
|---|---|---|---|---|---|---|---|
| t1 | yolo26l, 4 | 2% | off | 9.1 GB | 7.1 / 5.1 GB | n/a | OK |
| t2 | yolo26l, 6 | 2% | off | | 7.9 / 7.6 GB | n/a | Card full |
| t3 | yolo26l, 8 | 2% | off | | 10.1 GB torch | n/a | Overflow, 4x slower |
| t4 | yolo26s, 16 | 2% | off | | 8.2 GB torch | n/a | At the limit |
| t5 | yolo26s, 32 | 2% | off | | 15.9 GB torch | n/a | Overflow |
| t6 | yolo26s, 8 | 2% | off | | 4.8 / 4.2 GB | 49 img/s | OK |
| t7 | yolo26s, 12 | 2% | off | | 6.9 / 6.3 GB | 53 img/s | OK |
| t8 | yolo26l, 4 | full | off | 12.8 GB | 5.5 / 4.9 GB | 41 min + 2 min val | OK |
| t8r | yolo26l, 4 | full | off | | | | Killed in epoch 2, resumed at epoch 2 |
| t9 | yolo26s, 12 | full | ram | 28.5 GB | 6.5 / 5.9 GB | 12 min + 1 min val | OK |
| e2e-b | all three stages | 2%, 1 epoch | off | 13.6 GB | 7.5 / 6.8 GB | n/a | OK, but teachers proposed 0 labels |
| e2e-c | all three stages, from the Nov 13 checkpoint | 2%, 1 epoch | off | 13.4 GB | 7.5 / 6.8 GB | n/a | OK, label set written and used |

Logs are in `runs/logs/memtest-*.log`, with RAM and VRAM samples in the matching
`*.stats.csv`. The test checkpoints and containers were deleted after the tests.

## Full run

**First attempt (2026-11-16, stopped after 11 epochs).** Stage 1 ran at about 43 minutes
per epoch with 8.2 GB of RAM and 5.4 to 5.9 GB on the card, and reached mAP50-95 0.73.
It was stopped on purpose because the synthetic dataset was wrong (see below). Its
checkpoints and log are kept as `angelicam/full_unlabeled-backgrounds_2026-11-16/` and
`runs/logs/yolo-train.unlabeled-backgrounds.log`; both can be deleted.

**Second attempt (2026-11-16).** Restarted from scratch with the corrected synthetic
dataset, as container `yolo-train` writing to `angelicam/full/`. Regenerating the 50,000
synthetic images takes about 1 hour 45 minutes before training starts.

Expected total is about three days: pretrain 36 hours, the three pseudo-label teachers up
to 15 hours (estimate, not measured at full size), final training about 23 hours. The
corrected synthetic images are larger on average than before (many are 1280x720 or
1920x1080 instead of 640x640), so epochs may take longer than the 43 minutes measured in
the first attempt. Early stopping (patience 20) can shorten any stage.

## Synthetic dataset problems found and fixed

The synthetic set is 50,000 images (42,500 for training, about 70% of the training set),
made by pasting rendered plates onto real photos.

| Problem | Effect | Fix |
|---|---|---|
| Only the pasted plates were labelled; the labels the background photo already had were dropped | Every real plate in a background was trained as "not a plate" | The background's labels (all classes) are written with the pasted plates' boxes. A box is dropped only if a pasted plate covers more than half of it |
| Backgrounds were searched with the pattern `train/images` and capped at 500 | Only `Brazil.Plates.Detector` was used, each photo about 100 times. `UFPR-ALPR` and `RodoSol-ALPR` use `images/training` and never matched; the cap was reached before `FACE.DETECTION` was visited | Backgrounds are read from the `train` list in `data.yaml`: all four datasets, about 18,000 photos, each dataset picked equally often |
| Synthetic val and test images used the same training photos | Validation on synthetic images was not independent of training | Each split takes its backgrounds from the same split of the real datasets |
| The search pattern also matched `datasets/synthetic_plates/train/images` | A regeneration would have pasted plates onto already-synthetic images | Only directories listed in `data.yaml` are used |

The code is in `synthetic_plates/dataset_builder.py` and `stage_generate_synthetic` in
`training/train_pipeline.py`. The generation log prints how many background labels were
kept and how many were dropped as covered.

## MLflow system metrics

CPU, RAM, disk and GPU usage were not in MLflow because MLflow only records them when
`MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true` is set, and nothing set it. The packages it
needs (`psutil`, `nvidia-ml-py`) were already in the image. The variable is now set in
`run_full.sh` and in the three compose files; the metrics appear under the run's
"System metrics" tab as `system/cpu_utilization_percentage`,
`system/system_memory_usage_megabytes`, `system/gpu_0_memory_usage_megabytes`,
`system/gpu_0_utilization_percentage` and others, sampled every 10 seconds.

The memory figure is for the whole WSL VM, not only the container.

## Crashes and resume

Checkpoints (`last.pt`, `best.pt`) are written every epoch to the mounted `angelicam/`
folder. `run_full.sh` always passes `--resume`, so running it again continues an
interrupted stage from its `last.pt` and reuses stages that already finished. Resume was
tested by killing t8 in its second epoch.

To tell a RAM kill from a VRAM failure:

```bash
docker inspect yolo-train --format '{{.State.OOMKilled}}'
tail -50 runs/logs/yolo-train.log
```

- `true`: the container hit the 44 GB RAM limit.
- `false` and `CUDA error: out of memory` in the log: VRAM. Re-run with `FINAL_BATCH=8`
  (final stage) or `BATCH=2` (`yolo26l` stages).

## Code changes made for this

- `training/train_yolov26.py`: `train()` takes `cache`, `imgsz`, `fraction` and `resume`.
  Fixed `log_model_to_mlflow`, which raised `UnboundLocalError` on `mlflow` after every
  training.
- `training/train_pipeline.py`: new flags `--final-batch-size`, `--teacher-epochs`,
  `--imgsz`, `--workers`, `--cache {ram,disk,off}`, `--fraction`, `--resume`.
- `pseudo_labeling/backends.py`: the teachers were hardcoded to batch 16 with RAM cache
  (19.3G torch, 8 s/it on this card). They now take their settings from the pipeline
  through `PL_TEACHER_*` environment variables.
- `synthetic_plates/dataset_builder.py`, `training/train_pipeline.py`: the synthetic
  dataset fixes described above.
- `run_full.sh`: the launcher. Overrides: `BATCH`, `FINAL_BATCH`, `TRAIN_NAME`,
  `TRAIN_PROJECT`, `LABEL_SETS_DIR`, `EXTRA_ARGS`, `MLFLOW_TRACKING_URI`.

## Things to know

- WSL exports `NAME` as the hostname (`bh02`), so scripts must not use `$NAME` as an
  input variable. `run_full.sh` uses `TRAIN_NAME` for that reason.
- `--base-model` does not change the pretrain model; `yolo26l` is hardcoded for stage 1.
- One-epoch teachers are too weak to propose any pseudo-labels. With no label set, the
  final stage falls back to `data.yaml` without failing, so check the log for
  `proposed=` and `merged=True` after stage 2.
- The `Python.h` message from triton at startup is harmless.
