# Experiment ideas

Ideas to test after the current full run (started 2026-11-16 on bh02). None of them are
implemented yet unless marked otherwise. Each one should be compared against the current
run as the baseline, on the **real** validation and test images only.

## Already changed (2026-11-17, in the current run)

- **Final stage validates on real images only.** Synthetic images stay in its training
  split, but no longer in val/test, so early stopping and `best.pt` reflect real photos.
- **Teachers capped at 30 epochs** (`--teacher-epochs 30`, was 100). 30 is a guess: check
  each teacher's `results.csv` for the epoch where mAP50-95 peaks and adjust.

Problems found in the 2026-11-16 run (pseudo-labels written into val/test, duplicate
car/motorcycle teachers, teachers barely improving, a run lost to an MLflow outage) are
scheduled as phase 1 of `IMPROVEMENT_PLAN.md`.

## Model selection and early stopping

| Idea | Why | How to test |
|---|---|---|
| Score each teacher on its own class only | Teachers are judged by mAP50-95 over all 4 classes, so a better face teacher can look worse because of the other classes | Custom fitness in the teacher training, or validate with `classes=[target]`; compare pseudo-label precision |
| Validate the pretrain on real images only | A third of its val set is synthetic, which affects which epoch becomes the teacher's starting point | Same change as the final stage, applied to `stage_pretrain` |
| Shorter pretrain | mAP50-95 was flat from about epoch 25 (0.855) to 40 (0.855, peak 0.859 at 33); the last 20 epochs cost about 15 hours | 30 epochs, or patience 10; compare teacher quality |

## Is each stage worth its cost?

| Idea | Why | How to test |
|---|---|---|
| Teachers from COCO `yolo26l.pt` instead of the pretrain | The pretrain's main gain is on plates (synthetic data), but plates are not pseudo-labelled; the teachers are for face, car and motorcycle | Run stage 2 both ways, compare the pseudo-labels against a hand-checked sample |
| Distil the pretrain into the final model | The final `yolo26s` starts from COCO weights and gets nothing directly from the 36-hour pretrain | Ultralytics shows a `distill_model` training argument; check what it supports before relying on it |
| Final model without pseudo-labels | Measures how much stage 2 actually adds | Final stage on the original `data.yaml` |

## Synthetic data

| Idea | Why | How to test |
|---|---|---|
| Fewer synthetic images | 42,500 synthetic images are about 70% of the training set | 0 / 10k / 25k / 42.5k, measured on real val |
| Occlusion rule | A background box is dropped when a pasted plate covers more than 50% of it | 30% and 70% |
| Background balance | Each of the 4 datasets is picked equally often, so the 1,207 `Brazil.Plates.Detector` photos are reused about 9 times each | Weight by dataset size instead |

## Training settings

| Idea | Why | How to test |
|---|---|---|
| Lighter geometric augmentation | `degrees=45` and `shear=45` are very strong for plates, which are rarely rotated that much | `degrees=10`, `shear=5` |
| Smaller hue shift | `hsv_h=0.3` changes colours a lot, and plate colour carries meaning (Mercosul blue band, red commercial plates) | `hsv_h=0.015` (Ultralytics default) |
| Larger image size | Plates are small in the 1920x1080 RodoSol and UFPR images | `imgsz=960`; needs the 16 GB machine or a smaller batch |
| Pseudo-label confidence threshold | Default 0.5 for every class | 0.3 / 0.5 / 0.7, check precision on a hand-checked sample |

## Evaluation

- Report mAP per dataset (Brazil, FACE, UFPR, RodoSol) and per class on the real test
  split, not just the overall number, so a gain on one dataset can't hide a loss on another.
- Keep one fixed hand-checked sample of pseudo-labels to measure their precision.

## What gets reused between runs

With `./run_full.sh` (it always passes `--resume`), a finished step is reused and an
unfinished one continues from its last checkpoint:

| Step | Stored in | Reused when | Forced to redo with |
|---|---|---|---|
| Synthetic dataset | `datasets/synthetic_plates/` | Always, if present | `--regenerate-synthetic` |
| Pretrain | `<project>/pretrain/` | Same project | New `TRAIN_PROJECT`, or delete the folder |
| Teachers | `<project>/teachers/` | Same project | New project, or `--regenerate-pseudo` |
| Pseudo-label set | `datasets/label_sets/` (shared by all projects) | Always, if present | `--regenerate-pseudo` |
| Final model | `<project>/<name>/` | Same project and name | New `TRAIN_PROJECT` |

For experiments: an experiment that only changes the final stage should reuse the
synthetic set, the pretrain and the label set. Today that is awkward: `run_full.sh` always
passes `--pretrain`, and the pretrain folder belongs to the project, so a new
`TRAIN_PROJECT` would retrain the pretrain (36 hours) for nothing. The improvement plan
should add a way to skip stages or point at an existing pretrain/label set first.
