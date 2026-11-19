"""MLflow bookkeeping for the training pipeline.

Keeps runs small and comparable (see ``training/IMPROVEMENT_PLAN.md``, phase 2):

* one **parent run** per pipeline launch, holding the description (purpose, choices,
  commit); every stage's run (pretrain, teachers, final) is nested under it;
* **tags** on every run: commit, uncommitted-changes flag, idea, baseline, stage, host, GPU;
* **datasets** recorded with ``mlflow.log_input`` as metadata only (name, source, digest),
  plus a few-KB ``dataset_manifest.json``;
* **one copy of each model**: only ``best.pt`` is uploaded (Ultralytics also uploaded
  ``last.pt``); final models are registered in the Model Registry.

Everything here is best effort: a tracking problem is logged and never stops training.

Settings travel between stages (and into the pseudo-labeling subprocess code) through
environment variables, so ``train_yolov26.train`` needs no extra arguments:

``MLFLOW_PARENT_RUN_ID``   parent run of the current pipeline launch
``PIPELINE_TAGS``          JSON dict of tags copied onto every stage run
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path

import yaml

try:
    import mlflow
except ImportError:  # pragma: no cover - mlflow is in requirements
    mlflow = None

logger = logging.getLogger(__name__)

PARENT_ENV = "MLFLOW_PARENT_RUN_ID"
TAGS_ENV = "PIPELINE_TAGS"
DEFAULT_EXPERIMENT = "license-plate"
DEFAULT_REGISTERED_MODEL = "license-plate-detector"
PARENT_ID_FILE = "mlflow_parent_run_id"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def enabled() -> bool:
    """True when MLflow is installed and a tracking server is configured."""
    return mlflow is not None and bool(os.environ.get("MLFLOW_TRACKING_URI"))


# ---------------------------------------------------------------------------
# parent run
# ---------------------------------------------------------------------------
def start_pipeline_run(project: str, description: str, tags: dict[str, str]) -> str | None:
    """Create (or, for a resumed launch, reuse) the parent run of this pipeline launch.

    The parent run ID is kept in ``<project>/mlflow_parent_run_id`` so a launch that
    continues an interrupted one nests its stages under the same parent. All stage runs go
    to one experiment (``MLFLOW_EXPERIMENT_NAME``, default ``license-plate``) because
    MLflow only nests runs within an experiment.
    """
    if not enabled():
        return None
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)
    tags = {k: str(v) for k, v in tags.items() if v not in (None, "")}
    os.environ[TAGS_ENV] = json.dumps(tags)
    try:
        from mlflow import MlflowClient

        client = MlflowClient()
        experiment = mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
        id_file = Path(project) / PARENT_ID_FILE
        run_id = id_file.read_text().strip() if id_file.is_file() else ""
        if run_id:
            try:
                client.get_run(run_id)
                client.update_run(run_id, status="RUNNING")
                logger.info("Continuing MLflow parent run %s", run_id)
            except Exception:  # noqa: BLE001 - deleted or from another server
                run_id = ""
        if not run_id:
            run = client.create_run(
                experiment.experiment_id,
                run_name=Path(project).name,
                tags={**tags, "pipeline.role": "parent"},
            )
            run_id = run.info.run_id
            id_file.parent.mkdir(parents=True, exist_ok=True)
            id_file.write_text(run_id)
            logger.info("Created MLflow parent run %s", run_id)
        client.set_tag(run_id, "mlflow.note.content", description)
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
        os.environ[PARENT_ENV] = run_id
        return run_id
    except Exception as e:  # noqa: BLE001 - tracking must never stop training
        logger.warning("Could not set up the MLflow parent run: %s", e)
        return None


def end_pipeline_run(status: str = "FINISHED") -> None:
    """Mark the parent run finished (or failed)."""
    run_id = os.environ.get(PARENT_ENV)
    if not (enabled() and run_id):
        return
    try:
        from mlflow import MlflowClient

        MlflowClient().set_terminated(run_id, status=status)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not close the MLflow parent run: %s", e)


def describe_launch(args, defaults: dict, note: str, commit: str, dirty: bool) -> str:
    """Markdown description for the parent run: purpose, changed settings, commit."""
    changed = {
        key: value for key, value in sorted(vars(args).items())
        if key in defaults and value != defaults[key]
    }
    lines = [note.strip() or "_No purpose given (set EXPERIMENT_NOTE)._", ""]
    lines.append(f"- **Commit:** `{commit}`" + (" (with uncommitted changes)" if dirty else ""))
    if os.environ.get("EXPERIMENT_IDEA"):
        lines.append(f"- **Idea:** `{os.environ['EXPERIMENT_IDEA']}`")
    if os.environ.get("EXPERIMENT_BASELINE"):
        lines.append(f"- **Compared against:** run `{os.environ['EXPERIMENT_BASELINE']}`")
    lines += ["", "**Settings that differ from the defaults:**", ""]
    lines += [f"- `{key}` = `{value}`" for key, value in changed.items()] or ["- none"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def _resolve_root(data_yaml: str, cfg: dict) -> Path:
    root = Path(cfg.get("path", "."))
    if root.is_absolute():
        return root
    if root.exists():
        return root.resolve()
    return (Path(data_yaml).resolve().parent / root).resolve()


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            return Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
    return image.with_suffix(".txt")


def split_dirs(data_yaml: str, split: str) -> list[Path]:
    """Image directories of ``split`` in ``data_yaml``."""
    with open(data_yaml, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    root = _resolve_root(data_yaml, cfg)
    entries = cfg.get(split) or []
    if isinstance(entries, str):
        entries = [entries]
    return [Path(e) if Path(e).is_absolute() else (root / e) for e in entries]


def summarize_split(data_yaml: str, split: str) -> dict:
    """Digest plus per-directory image and box counts for one split.

    The digest covers each image's relative path and size and its label file's
    contents: any added, removed or relabelled image changes it. Image pixels are not
    read (that would take minutes); a changed image of the same size is not detected.
    """
    digest = hashlib.sha256()
    sources = {}
    for directory in split_dirs(data_yaml, split):
        images = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTS) \
            if directory.is_dir() else []
        boxes: Counter = Counter()
        for image in images:
            label = _label_path(image)
            text = label.read_text(errors="replace") if label.is_file() else ""
            digest.update(f"{image.relative_to(directory)}|{image.stat().st_size}|".encode())
            digest.update(text.encode())
            boxes.update(line.split()[0] for line in text.splitlines() if line.strip())
        name = "/".join(directory.parts[-3:])
        digest.update(name.encode())
        sources[name] = {"images": len(images), "boxes_per_class": dict(sorted(boxes.items()))}
    return {"digest": digest.hexdigest()[:16], "sources": sources}


def log_datasets(data_yaml: str, run_id: str | None = None) -> dict:
    """Record train/val/test of ``data_yaml`` on the active run (metadata only)."""
    if not enabled():
        return {}
    summary = {}
    try:
        from mlflow.data.meta_dataset import MetaDataset

        for split, context in (("train", "training"), ("val", "validation"), ("test", "testing")):
            info = summarize_split(data_yaml, split)
            if not info["sources"]:
                continue
            summary[split] = info
            source = _dataset_source(split_dirs(data_yaml, split)[0])
            dataset = MetaDataset(source=source, name=f"{Path(data_yaml).stem}:{split}",
                                  digest=info["digest"])
            mlflow.log_input(dataset, context=context)
            mlflow.set_tag(f"data.{split}.digest", info["digest"])
        mlflow.log_dict({"data_yaml": data_yaml, **summary}, "dataset_manifest.json")
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not record the datasets in MLflow: %s", e)
    return summary


def _dataset_source(path: Path):
    """A dataset source pointing at a local directory (only its path is stored)."""
    import warnings

    from mlflow.data.dataset_source_registry import resolve_dataset_source

    with warnings.catch_warnings():  # "can be interpreted in multiple ways": harmless
        warnings.simplefilter("ignore")
        return resolve_dataset_source(str(path))


# ---------------------------------------------------------------------------
# stage runs (Ultralytics callbacks)
# ---------------------------------------------------------------------------
def attach(yolo, stage: str, data_yaml: str) -> None:
    """Hook a stage's Ultralytics training into the pipeline's MLflow bookkeeping.

    Ultralytics adds its MLflow callbacks when the trainer is created, after the model's
    own callbacks, so the wiring is done at ``on_pretrain_routine_start``: it appends a
    hook that runs after Ultralytics has opened its run (tags, parent, datasets), and
    replaces Ultralytics' end-of-training upload (``best.pt`` *and* ``last.pt``) with one
    that uploads ``best.pt`` only.
    """
    if not enabled():
        return

    def wire(trainer):
        callbacks = trainer.callbacks
        starts = callbacks.get("on_pretrain_routine_end", [])
        for i, f in enumerate(starts):
            if _is_ultralytics_mlflow(f, "on_pretrain_routine_end"):
                starts[i] = _without_param_conflicts(f)
        ours = [f for f in callbacks.get("on_train_end", []) if _is_ultralytics_mlflow(f)]
        for f in ours:
            callbacks["on_train_end"].remove(f)
        if ours:
            callbacks["on_train_end"].append(_upload_best_only)
        callbacks["on_pretrain_routine_end"].append(
            lambda t: _tag_stage_run(t, stage, data_yaml)
        )

    yolo.add_callback("on_pretrain_routine_start", wire)


def _is_ultralytics_mlflow(func, name: str = "on_train_end") -> bool:
    return getattr(func, "__module__", "").endswith("callbacks.mlflow") and \
        getattr(func, "__name__", "") == name


def _without_param_conflicts(start_callback):
    """Ultralytics' run start, logging only parameters the run doesn't have yet.

    A resumed training continues its earlier MLflow run, and Ultralytics logs all its
    parameters again; some differ on resume (``model`` is now ``last.pt``). MLflow
    rejects changed parameters, and Ultralytics then stops tracking the rest of the
    training. The parameters recorded at the original start are kept.
    """
    def start(trainer):
        original = mlflow.log_params

        def log_new_params(params, *args, **kwargs):
            run = mlflow.active_run()
            existing = mlflow.get_run(run.info.run_id).data.params if run else {}
            new = {k: v for k, v in params.items() if k not in existing}
            if new:
                original(new, *args, **kwargs)

        mlflow.log_params = log_new_params
        try:
            start_callback(trainer)
        finally:
            mlflow.log_params = original

    start.__name__ = start_callback.__name__
    start.__module__ = start_callback.__module__
    return start


def _tag_stage_run(trainer, stage: str, data_yaml: str) -> None:
    if not getattr(trainer, "_mlflow_active", False) or mlflow.active_run() is None:
        return
    try:
        tags = json.loads(os.environ.get(TAGS_ENV, "{}"))
        tags["stage"] = stage
        parent = os.environ.get(PARENT_ENV)
        if parent:
            tags["mlflow.parentRunId"] = parent
            tags.setdefault("mlflow.note.content", f"Stage `{stage}` of parent run `{parent}`.")
        try:
            import torch
            if torch.cuda.is_available():
                tags["gpu"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            pass
        mlflow.set_tags(tags)
        if not getattr(trainer, "_datasets_logged", False):
            log_datasets(data_yaml)
            trainer._datasets_logged = True
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not tag the MLflow run: %s", e)


def _upload_best_only(trainer, status: str = "FINISHED") -> None:
    """Ultralytics' end-of-training upload, minus ``last.pt`` (kept on disk for resume)."""
    from ultralytics.utils.callbacks import mlflow as ul_mlflow

    if status == "FINISHED" and getattr(trainer, "_mlflow_active", False):
        try:
            if Path(trainer.best).is_file():
                mlflow.log_artifact(str(trainer.best), artifact_path="weights")
            for f in Path(trainer.save_dir).glob("*"):
                if f.suffix in {".png", ".jpg", ".csv", ".yaml"}:
                    mlflow.log_artifact(str(f))
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not upload results to MLflow: %s", e)
    # Close the run the way Ultralytics does; with _mlflow_active off it skips its upload.
    trainer._mlflow_active = False
    ul_mlflow.on_train_end(trainer, status=status)


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------
def register_model(run_id: str, description: str = "", tags: dict | None = None,
                   name: str | None = None) -> str | None:
    """Register ``weights/best.pt`` of ``run_id`` as a new model version.

    The new version gets the ``candidate`` alias; ``champion`` is moved by hand (or by
    the benchmark) once it has been compared. Returns the version number.
    """
    if not enabled():
        return None
    name = name or os.environ.get("MLFLOW_REGISTERED_MODEL", DEFAULT_REGISTERED_MODEL)
    try:
        from mlflow import MlflowClient

        client = MlflowClient()
        try:
            client.get_registered_model(name)
        except Exception:  # noqa: BLE001 - first version
            client.create_registered_model(
                name, description="YOLO detector for plates, faces, cars and motorcycles. "
                                  "Weights: best.pt (Ultralytics). Scores: see the benchmark tags.")
        source = client.get_run(run_id).info.artifact_uri.rstrip("/") + "/weights"
        version = client.create_model_version(
            name, source=source, run_id=run_id, description=description,
            tags={k: str(v) for k, v in (tags or {}).items()},
        )
        client.set_registered_model_alias(name, "candidate", version.version)
        logger.info("Registered %s version %s (alias @candidate)", name, version.version)
        return version.version
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not register the model: %s", e)
        return None
