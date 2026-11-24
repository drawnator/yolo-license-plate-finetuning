# Improvement plan

Goal: run the ideas in `EXPERIMENT_IDEAS.md` as comparable experiments, with every model
traceable in MLflow (what was changed, which code, which data) and scored on one fixed
benchmark, so any two models can be compared and downloaded later.

Order matters: phases 1 to 3 make results trustworthy and cheap; only then is it worth
running experiments (phase 4).

**Status (2026-11-24):** the phase 1 baseline (`angelicam/phase1-baseline-r2`, registry
version 2) finished and was benchmarked together with every earlier model, including the
GitHub releases V1 to v11 (see "Results" below and the dashboard). It scores **below** the
current champion, so it was not promoted.

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

## Results (benchmark v1, 2026-11-24)

All 16 models in `evaluation/models.yaml` were scored with the same benchmark; the full
table, per class and per dataset, is in `evaluation/results/benchmark_v1.csv` and the
dashboard. mAP50-95 on all four classes (models that detect only plate and face are
compared on those two in the dashboard):

| Model | Made | Overall | Plate | Face | Car | Motorcycle |
|---|---|---|---|---|---|---|
| v10 | 2026-07, first pseudo-label pipeline | **0.788** | 0.749 | **0.784** | 0.832 | 0.697 |
| v11 (registry v1, `@champion`) | 2026-11-18, pretrain + teachers + synthetic | 0.785 | **0.750** | 0.752 | 0.851 | 0.689 |
| no-pseudo-partial | 2026-11-19, no pseudo-labels, 38 epochs | 0.780 | 0.746 | 0.751 | 0.849 | 0.674 |
| v9 | 2026-06, real labels only | 0.778 | 0.745 | 0.743 | 0.815 | **0.727** |
| phase1-r2 (registry v2, `@candidate`) | 2026-11-21, phase 1 fixes | 0.769 | 0.725 | 0.739 | **0.855** | 0.671 |

What this says, and what it doesn't:

- **The phase 1 baseline is 0.016 below v11.** The loss is mostly plates (FACE images:
  0.44 against 0.51) and faces; cars improved slightly.
- **The training data was nearly the same; validation was not.** Both trained on about
  6,000 pseudo-labels in the train split (5,957 for phase1-r2, 5,938 for v11; v11's other
  6,865 were in val and test). But v11 was validated partly against teacher labels, and
  validation decides early stopping and which epoch becomes `best.pt`. phase1-r2
  early-stopped at epoch 38 (best epoch 18), while v11 trained 66 epochs (best 46).
  Training length is therefore the most likely cause, not the pseudo-labels themselves.
- **Pseudo-labels did not help here.** The accidental run without pseudo-labels, also
  stopped at epoch 38, scored 0.780: above phase1-r2 and close to v11.
- **v10 from July is still the best overall** (0.788), and also on faces. The
  36-hour pretrain and synthetic plates have not beaten it on human-labelled test data;
  their gain is on cars.
- **On the FACE dataset, newer models are worse, and synthetic plates are a suspect.**
  Every recent model scores below the May models there; V3 (nano, plate + face only) is
  best at 0.705. The three models trained with synthetic plates (run-2026-08-16, v11 and
  phase1-r2) are the bottom three, 0.59 to 0.63. The loss is plates in face photos: 0.44
  to 0.51 against 0.60 for V3. This doesn't prove the synthetic plates are the cause, but
  it is worth testing (the synthetic-data experiment of phase 4).
- **The noise is unknown.** Each configuration was trained once. Differences of about
  0.01 between the top five may be within run-to-run variation.

Next steps, in order:

1. **Measure the noise:** train the phase1-r2 configuration a second time with another
   seed (final stage only, about 10 hours). Without this, no difference under about
   0.02 can be trusted.
2. **Early stopping:** with human labels only, validation peaked at epoch 18 and the run
   stopped at 38 (v11: 46 of 66). Either the final stage now stops too early, or the
   smaller real-only validation set is noisy. Try `patience=40`, or a fixed number of
   epochs.
3. **No pseudo-labels, trained to the end** (`--no-pseudo-labels`): the partial run
   suggests stage 2 is not worth its cost.
4. Only then the experiments of phase 4.

Promotion is a manual decision: v11 stays `@champion`. v10 is not in the registry; to
make it the champion, register its MLflow run (`0d28cb9d…`) as a new version first.

## Dashboard

`python3 -m evaluation.dashboard` (http://localhost:8000) compares every benchmarked
model:

- ranked bars;
- a table of differences to a chosen baseline, per class and per dataset;
- accuracy against speed, and score over time;
- a sortable table with links to MLflow and the GitHub releases.

The metric, the set of classes and the set of datasets can be changed (for example
FACE only), and "Compare two scores" plots any two scores against each other: overall,
a class, a dataset, or one class in one dataset (default: face against plate). A model is ranked only when it can
detect every selected class, so the plate + face models V1 to v5 appear with "Plate +
face".

It reads `evaluation/results/benchmark_v1.csv`. To add a model:
1. Add it to `evaluation/models.yaml`.
2. Run the benchmark with `--catalog evaluation/models.yaml` (models already scored are
   skipped, by the hash of their weights).
3. Run `python -m evaluation.export_results`.

The commands are in the README, under "Evaluation".

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

Each weights file is evaluated once per benchmark version (results are cached by its
SHA-256); later comparisons reuse the stored result. Besides registry references, the
benchmark takes `release:<tag>` (a GitHub release), `runs:/<id>`, a file, or
`--catalog evaluation/models.yaml` for every listed model. The scores also appear as `bench.v1.*` tags on the training run and the registry
version. To promote a model: set the `champion` alias on its version in the MLflow UI.

**Download a model:** `python -m evaluation.model_store license-plate-detector@champion`
(prints the path of the downloaded `best.pt`; same image and variables as above).
