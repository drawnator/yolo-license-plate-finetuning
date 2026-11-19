"""
YOLOv26n Fine-tuning Script for License Plate Detection
=======================================================

Hardware Requirements:
    Minimum : NVIDIA GPU with 10 GB VRAM (e.g. RTX 3080 / RTX 4070)
    Recommended: NVIDIA GPU with 24 GB VRAM (e.g. RTX 4090 / A6000)
    RAM      : 32–64 GB system RAM
    Storage  : 30 GB free disk space

Installation:
    pip install ultralytics>=8.3.0
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import mlflow
except ImportError:
    mlflow = None

# import albumentations as A

from ultralytics import YOLO

try:
    from training import tracking
except ImportError:  # run as a plain script from inside training/
    import tracking  # type: ignore

logger = logging.getLogger(__name__)
# https://docs.ultralytics.com/guides/yolo-data-augmentation#custom-albumentations-transforms-augmentations
# custom_transforms = [
#     A.Blur(blur_limit=7, p=0.5),
#     A.GaussNoise(p=0.3),
#     A.CLAHE(clip_limit=4.0, p=0.5),
#     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    
# ]

def _get_git_hash() -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return os.environ.get("GIT_COMMIT_HASH", "unknown")


def _check_environment():
    """Validate environment before starting training."""
    if not Path("datasets").exists():
        raise RuntimeError(
            "No datasets/ directory. Run `python -m prep_dataset` first."
        )

    # Disk space warning
    free_gb = shutil.disk_usage(".").free / (1024**3)
    if free_gb < 5:
        logger.warning("Less than 5GB free disk space — training may fail")

    # CUDA availability
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA is available")
        else:
            logger.warning("CUDA is not available — training will use CPU")
    except ImportError:
        logger.warning("torch not installed, assuming CPU-only")


def train(
    data: str = r"./data.yaml",
    model: str = "yolo26s.pt",
    batch_size: int = 16,
    epochs: int = 100,
    device: str = "0",
    project: str = "angelicam",
    name: str = "yolov26_license_plate",
    patience: int = 20,
    workers: int = 8,
    cache: bool | str = True,
    imgsz: int = 640,
    fraction: float = 1.0,
    resume: bool = False,
    **overrides,
):
    """
    Fine-tune YOLOv26 on a license plate dataset.

    Args:
        data: Path to the dataset YAML file.
        model: Pre-trained model to start from.
        batch_size: Training batch size (reduce if OOM, or set -1 for auto).
        epochs: Number of training epochs.
        device: CUDA device index or 'cpu'.
        project: Output directory for training runs.
        name: Name for this training run.
        patience: Early-stopping patience (epochs without improvement).
        workers: Number of data-loading workers.
        cache: Image cache mode: True/"ram", "disk", or False (lowest RAM use).
        imgsz: Training image size.
        fraction: Fraction of the training set to use (for smoke tests).
        resume: Continue the newest unfinished run of ``name`` under ``project``
            from its ``last.pt`` instead of starting over. A run that already
            finished is reused as-is.
        **overrides: Any other Ultralytics training argument (``lr0``, ``optimizer``,
            ``degrees``, ...). Applied last, so they replace the defaults below.
            Ignored on resume, where the checkpoint's settings apply.

    Returns:
        Ultralytics training results.
    """
    # --- Input validation ---
    data_path = Path(data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data}")

    if not model.lower().endswith(".pt"):
        raise ValueError(
            f"Expected a .pt model file, got '{model}'. "
            f"Valid examples: yolo26n.pt, yolo26s.pt, yolo26m.pt"
        )

    if not Path(model).exists():
        logger.info(f"[info] {model} not found locally, will be downloaded by ultralytics")

    if not (1 <= batch_size <= 256):
        raise ValueError(f"batch_size must be between 1 and 256, got {batch_size}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if patience < 0:
        raise ValueError(f"patience must be non‑negative, got {patience}")
    if workers < 0:
        raise ValueError(f"workers must be non‑negative, got {workers}")

    # Normalize device string
    if device != "cpu":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(f"device={device} but CUDA unavailable, falling back to cpu")
                device = "cpu"
        except ImportError:
            device = "cpu"

    # Idempotent output directory creation
    project_path = Path(project).resolve()
    project_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training outputs will be saved to: {project_path}")

    # A tracking-server hiccup at start-up makes Ultralytics train the whole run untracked.
    _wait_for_mlflow()

    # --- Resume an interrupted run if asked ---
    resume_from = _find_last_checkpoint(project_path, name) if resume else None
    if resume_from is not None:
        import torch
        from types import SimpleNamespace

        epoch = torch.load(resume_from, map_location="cpu", weights_only=False).get("epoch", -1)
        if epoch is None or epoch < 0:  # Ultralytics marks finished runs with epoch=-1
            logger.info("Run %s already finished — reusing it", resume_from.parent.parent)
            return SimpleNamespace(save_dir=resume_from.parent.parent)
        logger.info("Resuming from %s (last completed epoch index: %d)", resume_from, epoch)
        # Continue the MLflow run of the interrupted attempt instead of opening a new one.
        # MLflow resumes the run named in MLFLOW_RUN_ID; it is removed again afterwards so
        # later stages (teachers, final model) still get runs of their own.
        previous_run = _previous_mlflow_run(project_path, resume_from.parent.parent.name, name)
        if previous_run and not os.environ.get("MLFLOW_RUN_ID"):
            os.environ["MLFLOW_RUN_ID"] = previous_run
            logger.info("Continuing MLflow run %s", previous_run)
        else:
            previous_run = None
        try:
            # Only memory-related settings may change on resume; the rest comes from the checkpoint.
            resumed = YOLO(str(resume_from))
            tracking.attach(resumed, stage=name, data_yaml=str(data_path))
            results = resumed.train(
                resume=True,
                data=str(data_path),
                batch=batch_size,
                workers=workers,
                cache=cache,
                imgsz=imgsz,
                device=device,
            )
        except Exception as e:
            logger.error("Training failed: %s", e)
            raise
        finally:
            if previous_run:
                os.environ.pop("MLFLOW_RUN_ID", None)
        logger.info(f"Training complete. Results saved to: {results.save_dir}")
        return results
    if resume:
        logger.info("No checkpoint to resume under %s/%s* — starting fresh", project_path, name)

    # --- Start training ---
    logger.info(f"Loading model {model}...")
    yolo = YOLO(model)
    tracking.attach(yolo, stage=name, data_yaml=str(data_path))

    try:
        #https://docs.ultralytics.com/modes/train#train-settings
        settings = dict(
            data=str(data_path),
            batch=batch_size,
            epochs=epochs,
            device=device,
            project=str(project_path),
            name=name,
            patience=patience,
            workers=workers,
            verbose=False,
            exist_ok=False,
            cache=cache,
            imgsz=imgsz,
            fraction=fraction,
            cls_pw=0.3,
            # License‑plate‑specific augmentation
            hsv_h=0.3,
            close_mosaic=10,
            degrees=45.0,
            fliplr=0.5,
            flipud=0.01,
            mosaic=1.0,
            multi_scale=0.25,
            shear=45,
            perspective=0.001,
            cutmix=0.1,
            mixup=0.1,
            # augmentations=custom_transforms,
            copy_paste=0.2,
        )
        settings.update(overrides)
        results = yolo.train(**settings)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user. Partial checkpoint may be in: %s", project_path)
        sys.exit(1)
    except Exception as e:
        logger.error("Training failed: %s", e)
        raise

    logger.info(f"Training complete. Results saved to: {results.save_dir}")

    return results


def _find_last_checkpoint(project_path: Path, name: str) -> Path | None:
    """Return the newest ``last.pt`` among runs ``name``, ``name2``, ... under ``project_path``."""
    candidates = list(project_path.glob(f"{name}*/weights/last.pt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _wait_for_mlflow(attempts: int = 20, delay_s: float = 30.0) -> bool:
    """Wait until the MLflow tracking server answers (up to ``attempts * delay_s``).

    Returns True when it answers or no remote server is configured. If it never answers,
    logs an error and returns False: training then goes ahead, but untracked, since
    stopping a multi-day pipeline for a tracking outage would cost more.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if mlflow is None or not uri.startswith(("http://", "https://")):
        return True
    import time
    import urllib.request

    # The MLflow client's own user agent: the server's Cloudflare front answers 403 to
    # Python's default one.
    request = urllib.request.Request(
        uri.rstrip("/") + "/health",
        headers={"User-Agent": f"mlflow-python-client/{mlflow.__version__}"},
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status == 200:
                    if attempt > 1:
                        logger.info("MLflow server is reachable again")
                    return True
                problem = f"HTTP {response.status}"
        except Exception as e:  # noqa: BLE001 - network errors of any kind
            problem = str(e)
        logger.warning("MLflow server not reachable (%s), attempt %d/%d; retrying in %.0f s",
                       problem, attempt, attempts, delay_s)
        time.sleep(delay_s)
    logger.error("MLflow server still unreachable after %d attempts: this training will NOT "
                 "be tracked in MLflow", attempts)
    return False


def _previous_mlflow_run(project_path: Path, *run_names: str) -> str | None:
    """Return the ID of the newest MLflow run of this training, or ``None``.

    Ultralytics names the experiment after the project path (unless
    ``MLFLOW_EXPERIMENT_NAME`` is set) and the run after the run directory. Any lookup
    failure (no tracking server, no such run) returns ``None``: a new run is then opened.
    """
    if mlflow is None or not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    try:
        experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME") or str(project_path)
        # Within a pipeline launch, only this launch's runs (same parent) are candidates.
        parent = os.environ.get(tracking.PARENT_ENV)
        parent_filter = f" and tags.mlflow.parentRunId = '{parent}'" if parent else ""
        for run_name in dict.fromkeys(n for n in run_names if n):
            runs = mlflow.search_runs(
                experiment_names=[experiment],
                filter_string=f"attributes.run_name = '{run_name}'" + parent_filter,
                order_by=["attributes.start_time DESC"],
                max_results=1,
                output_format="list",
            )
            if runs:
                return runs[0].info.run_id
    except Exception as e:  # noqa: BLE001 - never block a resume on MLflow
        logger.warning("Could not look up the previous MLflow run (%s); a new one will be opened", e)
    return None


def best_weights_of(results) -> Path:
    """Return the path to ``best.pt`` for a completed training run."""
    return Path(results.save_dir) / "weights" / "best.pt"


def export_model(weights_path: Path, fmt: str = "coreml") -> Path | None:
    """Export best weights to ``fmt`` (default CoreML .mlpackage). Returns the export path.

    Returns ``None`` (and logs) on failure so callers can continue.
    """
    logger.info("Exporting %s to %s...", Path(weights_path).name, fmt)
    try:
        model = YOLO(str(weights_path))
        exported_path = Path(model.export(format=fmt, nms=True))
        logger.info("Exported to: %s", exported_path)
        return exported_path
    except Exception as e:  # noqa: BLE001 - export is best-effort
        logger.error("Export to %s failed: %s", fmt, e)
        return None


def log_model_to_mlflow(
    weights_path: Path,
    exported_path: Path | None = None,
    *,
    params: dict | None = None,
    artifacts: list | None = None,
    data_yaml: str | None = None,
    register: bool = False,
    register_description: str = "",
    register_tags: dict | None = None,
) -> None:
    """Add proper model logging to the MLflow run Ultralytics already created.

    Ultralytics' ``yolo.train`` auto-logs params/metrics to MLflow when the tracking env
    vars are set (as in docker-compose) and ends the run when training finishes. Rather than
    open a competing run, this **resumes that same run** (via ``mlflow.last_active_run``) and
    only increments it with the trained model, the CoreML export, and any extra ``artifacts``
    (e.g. a zipped self-supervised label set). If no prior run is found it falls back to a
    new run so standalone use still logs something.

    The model is stored once, as ``weights/best.pt`` (the file Ultralytics and the
    inference scripts load). A second pickled copy via ``mlflow.pytorch`` only doubled
    the run's size. With ``register=True`` that file becomes a new version in the Model
    Registry (see ``training/tracking.py::register_model``).

    If ``data_yaml`` is provided, its contents are logged as ``data_yaml_used.yaml`` so the
    exact dataset configuration used for training is preserved in the run.
    """
    if mlflow is None:
        logger.warning("mlflow not installed, skipping model logging")
        return

    try:
        # Resume the run Ultralytics created during training (keep its auto-logged data).
        last = mlflow.last_active_run()
        run_id = last.info.run_id if last is not None else None

        with mlflow.start_run(run_id=run_id):
            for key, value in (params or {}).items():
                mlflow.log_param(key, value)

            # Log the training data.yaml as a reference artifact
            if data_yaml and Path(data_yaml).exists():
                mlflow.log_artifact(data_yaml, artifact_path="config")
                logger.info("Logged training data.yaml to MLflow: %s", data_yaml)

            # best.pt is already in the run (weights/best.pt, uploaded at the end of
            # training); upload it here only if that did not happen.
            client = mlflow.MlflowClient()
            if run_id is None or not any(
                f.path.endswith("best.pt") for f in client.list_artifacts(run_id, "weights")
            ):
                mlflow.log_artifact(str(weights_path), artifact_path="weights")

            if exported_path is not None and Path(exported_path).exists():
                mlflow.log_artifact(str(exported_path), artifact_path="coreml")

            for art in artifacts or []:
                if art and Path(art).exists():
                    mlflow.log_artifact(str(art))
                    logger.info("Logged artifact to MLflow: %s", art)
        if register:
            run_id = run_id or mlflow.last_active_run().info.run_id
            tracking.register_model(run_id, description=register_description,
                                    tags=register_tags)
    except Exception as e:  # noqa: BLE001 - logging must never fail the run
        logger.error("MLflow model logging failed: %s", e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    _check_environment()
    results = train()

    # Export + log the trained model to MLflow (kept out of train() so training and
    # export/logging can be composed independently, e.g. by the pseudo-labeling pipeline).
    best_weights = best_weights_of(results)
    if best_weights.exists():
        exported = export_model(best_weights)
        log_model_to_mlflow(
            best_weights,
            exported,
            params={"commit_hash": _get_git_hash()},
        )
    else:
        logger.warning("best.pt not found after training; skipping export")
