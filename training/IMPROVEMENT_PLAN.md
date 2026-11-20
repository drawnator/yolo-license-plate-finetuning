# Improvement plan

Goal: run the ideas in `EXPERIMENT_IDEAS.md` as comparable experiments, with every model
traceable in MLflow (what was changed, which code, which data) and scored on one fixed
benchmark, so any two models can be compared and downloaded later.

Order matters: phases 1 to 3 make results trustworthy and cheap; only then is it worth
running experiments (phase 4).

**Status (2026-11-20):** the first phase 1 baseline ran without pseudo-labels (item 3
below); it was kept as the "no pseudo-labels" data point and the baseline was restarted
with the old teacher settings.

**Status (2026-11-19):** phases 1 to 3 and the stage reuse of phase 4 are implemented;
see "How to use it" at the end. Not done yet: scoring each teacher on its own class
(phase 1, item 2; duplicate teachers are removed instead).

## Phase 1: fix what makes scores unreliable

1. **Pseudo-labels only in the training split.** Today stage 2 also writes teacher
   predictions into val and test (for example 304 car labels in FACE `valid`), so the final
   model is partly scored against another model. Write pseudo-labels for `train` only.
2. **One teacher per class, judged on its class.** Teachers 2 and 3 (car, motorcycle)
   trained on the same data with the same settings and came out identical. Either train
   once per dataset, or restrict each teacher to its class (`classes=[target]`) so its
   score and early stopping reflect that class.
3. ~~**Lower teacher learning rate.**~~ Tried and reverted (2026-11-20). With SGD at
   `lr0=0.001` for 15 epochs the teachers scored well on their own data (vehicles 0.81
   mAP50-95) but proposed **0 pseudo-labels**. The pretrain learned from datasets where
   cars are unlabelled, so it treats cars in Brazil, FACE and RodoSol images as
   background; the pretrain itself finds cars only in UFPR images. The old setting
   (`optimizer=auto`, lr 0.01, 30 epochs) retrains enough to undo that and produced
   12,803 labels. The teachers' own validation can't show this, because it only uses
   the dataset they train on. The pipeline now stops if no label set is produced
   instead of training without one. `TEACHER_LR0` is still available but opt-in.
4. **MLflow outages must not drop a run.** A single Cloudflare 522 at start-up made
   Ultralytics train teacher 2 untracked. Check the server before each stage and retry.

## Phase 2: tracking in MLflow

Currently a run stores 115 Ultralytics parameters, per-epoch metrics, system metrics and
the weights, but no commit, no dataset information, no description, and nothing in the
Model Registry (it is empty). The changes below add that while keeping each run small:
metadata and hashes, not copies of the data.

### Run layout
- **One parent run per pipeline launch,** with each stage (pretrain, teachers, final) as a
  child run. All parts of one experiment then sit together, and a resumed stage continues
  its own child run (already implemented for resumes).
- **Description** (`mlflow.note.content`, the run's "Description" field in the UI),
  written automatically on the parent run: the experiment's purpose (from a new
  `EXPERIMENT_NOTE` setting), what differs from the baseline, the key settings and the
  commit.

### Tags (searchable, a few bytes each)
| Tag | Value |
|---|---|
| `mlflow.source.git.commit` | Commit hash. MLflow fills this itself only when git is available, which it isn't in the image, so it is set from `GIT_COMMIT_HASH` |
| `git.dirty` | `true` if the code had uncommitted changes; computed on the host by the launcher |
| `experiment.idea` | Short key from `EXPERIMENT_IDEAS.md`, for example `aug-light` |
| `experiment.baseline` | Run ID of the run it is compared against |
| `stage` | `pretrain`, `teacher-1`, `final`, ... |
| `data.train.digest`, `data.val.digest` | Dataset hashes (below) |
| `label_set.id`, `synthetic.digest` | Which pseudo-label set and synthetic set were used |
| `host`, `gpu` | For example `bh02`, `RTX 4060 Ti 8GB`; explains speed differences between machines |

### Datasets (MLflow's dataset tracking, metadata only)
- For each split, `mlflow.log_input` with a small dataset record: name, source path and a
  **digest**, a hash of the list of image files plus the contents of their label files.
  If anything in the data changes, the digest changes. It is shown on the run's
  "Datasets" panel, and runs can be filtered by it.
- A `dataset_manifest.json` artifact (a few KB): images and boxes per source dataset and
  class, number of pseudo-labels per class, the synthetic generation settings (count,
  seed, overlap rule), and the label-set ID.

### Models
- **Log each model once.** Today the final stage stores `best.pt` and `last.pt` (via
  Ultralytics) plus a second pickled copy (via `mlflow.pytorch.log_model`). Keep only
  `best.pt` of each stage; drop `last.pt` (it stays on disk for resuming).
- **Register final models** in the Model Registry under one name (proposed:
  `license-plate-detector`). Each final run becomes a new version, with its description
  and tags copied over. Use the aliases `@champion` for the best model so far and
  `@candidate` for the one being evaluated. MLflow replaced the old stages
  (Staging/Production) with aliases.
- **Downloading:** a small `evaluation/get_model.py <name>@champion` (or a version
  number) that downloads `best.pt` into `models/`. Inference scripts take that path.
- **Pretrain and teacher weights** are logged in their child runs but not registered;
  they are inputs, not products.

Expected size per experiment: tens of KB of metadata, plus about 20 MB per final
`yolo26s` model, 110 MB for a `yolo26l` pretrain, and teachers only if kept.

## Phase 3: one fixed benchmark

Training-time validation is not a fair comparison: each run may use different data. A
separate benchmark is.

- **Benchmark data:** the real test split of the four datasets with human labels only:
  no synthetic images, no pseudo-labels. It is frozen, and its digest is recorded as
  `benchmark.version` (`v1`). Results are only compared within the same version.
- **`evaluation/benchmark.py <model>`**, where the model is a registry name and version,
  an alias, or a file:
  - mAP50, mAP50-95, precision and recall, overall, per class and per source dataset, so
    a gain in one place can't hide a loss in another;
  - speed in ms per image at a fixed `imgsz`;
  - fixed settings for every model: `imgsz=640` and the same confidence and IoU
    thresholds.
- **Results are logged to MLflow** in an evaluation run linked to the model (MLflow 3 can
  attach metrics to a model and a dataset), and are also written to the model version's
  tags. The registry list then shows the scores directly.
- **Comparison:** MLflow's run comparison view, plus `evaluation/compare_models.py`
  (already in the repo) changed to read benchmark results from MLflow and print a table.
- **Baseline:** the run finishing now (final model `angelicam/full/yolov26_license_plate`)
  is benchmarked first and becomes registry version 1, `@champion`.

## Phase 4: run the experiments cheaply

1. **Reuse finished stages.** Add options to point at an existing pretrain and label set
   and to skip stages (`--reuse-pretrain <path>`, `--label-set <id>`, `--stages final`).
   An experiment that only changes the final stage then costs at most about a day
   instead of three.
2. **Screen, then confirm.** First run `yolo26s` with `--fraction 0.3` and 30 epochs.
   Ideas that beat the baseline on the benchmark are run at full size. Use `yolo26n`
   only to check that a change runs end to end; its augmentation and learning-rate
   results don't carry over to larger models.
3. **One change per experiment,** tagged with its idea key and baseline, so each
   difference has one cause.
4. **Suggested order** (data problems first, since every later result depends on them):
   1. Phase 1 fixes, as a new full run: the new baseline.
   2. Lighter augmentation (`degrees`, `shear`, `hsv_h`).
   3. Amount of synthetic data (0 / 10k / 25k / 42.5k).
   4. Final model without pseudo-labels: is stage 2 worth its cost?
   5. Teachers from COCO weights instead of the pretrain: is stage 1 worth 36 hours?
   6. Larger `imgsz` on the 16 GB machine.

## Decisions (defaults taken 2026-11-19; change if needed)

- Registry model name: `license-plate-detector` (override with `MLFLOW_REGISTERED_MODEL`).
- Pretrain and teacher weights: their `best.pt` is uploaded to their stage run (one copy,
  no `last.pt`); they are not registered.
- Uncommitted code: allowed, tagged `git.dirty=true` and noted in the description.

## How to use it

**Start an experiment** (any of the variables can be left out):

```bash
EXPERIMENT_NOTE="Lighter geometric augmentation: degrees 10, shear 5" \
EXPERIMENT_IDEA=aug-light \
EXPERIMENT_BASELINE=<parent run id of the baseline> \
TEACHER_WEIGHTS=angelicam/full/pretrain/pretrain/weights/best.pt \
TRAIN_PROJECT=angelicam/exp-aug-light \
./run_full.sh
```

`TEACHER_WEIGHTS` skips the 36-hour pretrain; the synthetic set and the pseudo-label set
are reused automatically (both are shared between projects).

**In MLflow** (experiment `license-plate`): one parent run per launch, named after the
project folder, with the description and tags; the stage runs are nested under it. The
final model is registered as a new version of `license-plate-detector` with the alias
`@candidate`.

**Benchmark and compare:**

```bash
docker run --rm --gpus all --memory=16g --shm-size=8g \
  -e MLFLOW_TRACKING_URI=https://mlflow.guilherme.zip \
  -e MLFLOW_TRACKING_USERNAME -e MLFLOW_TRACKING_PASSWORD \
  -v "$PWD/datasets:/datasets" -v "$PWD/models:/models" -e MODELS_DIR=/models \
  --entrypoint python yolo-train:latest -m evaluation.benchmark \
  license-plate-detector@champion license-plate-detector@candidate
```

Each model is evaluated once per benchmark version; later comparisons reuse the stored
result. The scores also appear as `bench.v1.*` tags on the training run and the registry
version. To promote a model: set the `champion` alias on its version in the MLflow UI.

**Download a model:** `python -m evaluation.model_store license-plate-detector@champion`
(prints the path of the downloaded `best.pt`; same image and variables as above).
