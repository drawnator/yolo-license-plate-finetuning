"""Three-stage training pipeline: pretrain → pseudo-label → final train.

Stage 1 (optional): Generate synthetic plates dataset and pretrain a strong
    teacher model on real + synthetic data. This model will be used for
    pseudo-labeling.

Stage 2: Run pseudo-labeling with the pretrained model as teacher, producing
    a label set of self-supervised annotations.

Stage 3: Train the final model on real + pseudo-labeled data.

Usage:
    # Full three-stage pipeline (Docker entrypoint)
    python training/train_pipeline.py --synthetic 5000 --pretrain --base-model yolo26s.pt

    # Skip pretrain, use a specific weights file as teacher
    python training/train_pipeline.py --teacher-weights runs/pretrain/best.pt

    # Skip synthetic generation entirely
    python training/train_pipeline.py --base-model yolo26s.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _get_git_hash() -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    import subprocess

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


# Default directories
DEFAULT_DATA_YAML = "./data.yaml"
DEFAULT_DATASETS_ROOT = "./datasets"
PRETRAIN_DIR = "runs/pretrain"
SYNTHETIC_DATASET_DIR = "datasets/synthetic_plates"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )


def _inject_synthetic_into_data_yaml(
    source_yaml: str,
    synthetic_dir: str,
    output_yaml: str | None = None,
) -> str:
    """Add synthetic dataset paths to a copy of data.yaml and return the new path.

    Paths are computed relative to the YAML's ``path`` field (the dataset root),
    *not* relative to the YAML file's directory. This ensures YOLO resolves
    entries like ``synthetic_plates/train/images`` correctly against
    ``path: ./datasets`` → ``./datasets/synthetic_plates/train/images``.
    """
    with open(source_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    yaml_root = cfg.get("path", ".")

    # Normalize existing entries to lists
    for key in ("train", "val", "test"):
        val = cfg.get(key, [])
        if isinstance(val, str):
            val = [val]
        # Add synthetic split if it exists
        synthetic_split = f"{synthetic_dir}/{key}/images"
        if os.path.isdir(synthetic_split):
            # Compute path relative to the YAML's dataset root (e.g. ./datasets)
            rel = os.path.relpath(synthetic_split, yaml_root)
            if rel not in val:
                val.append(rel)
        cfg[key] = val

    out = output_yaml or source_yaml.replace(".yaml", "_with_synthetic.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info("Injected synthetic dataset into %s", out)
    return out


def stage_generate_synthetic(
    image_count: int,
    output_dir: str = SYNTHETIC_DATASET_DIR,
    seed: int = 42,
    force: bool = False,
) -> str:
    """Generate synthetic plates dataset using existing dataset images as backgrounds.

    Idempotent: skips generation if ``output_dir`` already contains a valid
    YOLO dataset structure (train/val/test with images and a data.yaml).
    Use ``force=True`` to always regenerate.

    Returns the path to the generated ``data.yaml``.
    """
    expected_yaml = os.path.join(output_dir, "data.yaml")
    if not force and _synthetic_dataset_valid(output_dir):
        logger.info(
            "Synthetic dataset already exists at %s — skipping generation. "
            "Use --regenerate-synthetic to force recreation.",
            output_dir,
        )
        return expected_yaml

    logger.info("=" * 60)
    logger.info("STAGE 1a: Generating synthetic plates dataset (%d images)", image_count)
    logger.info("=" * 60)

    # Collect background images from existing datasets
    bg_dir = _collect_backgrounds(DEFAULT_DATASETS_ROOT)

    if not bg_dir or not os.listdir(bg_dir):
        logger.warning(
            "No background images found in %s. Using solid-color backgrounds instead.",
            DEFAULT_DATASETS_ROOT,
        )
        bg_dir = _create_solid_backgrounds()

    from synthetic_plates.dataset_builder import build_dataset

    yaml_path = build_dataset(
        background_dir=bg_dir,
        output_dir=output_dir,
        image_count=image_count,
        plates_per_image=1,
        max_plates_per_image=3,
        val_split=0.1,
        test_split=0.05,
        seed=seed,
    )
    logger.info("Synthetic dataset ready: %s", yaml_path)
    return yaml_path


def _synthetic_dataset_valid(output_dir: str) -> bool:
    """Return True if ``output_dir`` contains a valid-looking YOLO dataset."""
    data_yaml = os.path.join(output_dir, "data.yaml")
    if not os.path.isfile(data_yaml):
        return False
    for split in ("train", "val", "test"):
        img_dir = os.path.join(output_dir, split, "images")
        lbl_dir = os.path.join(output_dir, split, "labels")
        if not os.path.isdir(img_dir) or not os.listdir(img_dir):
            return False
    return True


def _collect_backgrounds(datasets_root: str) -> str:
    """Collect a subset of training images from existing datasets to use as backgrounds.

    Returns path to a temporary directory with symlinks/copies of background images.
    """
    import tempfile

    bg_dir = os.path.join(tempfile.gettempdir(), "synthetic_backgrounds")
    os.makedirs(bg_dir, exist_ok=True)

    # Walk all train/images directories
    src = Path(datasets_root)
    if not src.exists():
        return bg_dir

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    collected = 0
    max_bg = 500  # cap backgrounds to avoid huge temp dirs

    for img_dir in src.rglob("train/images"):
        if not img_dir.is_dir():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() in image_exts:
                dst = os.path.join(bg_dir, img_path.name)
                if not os.path.exists(dst):
                    try:
                        os.link(img_path, dst)  # hardlink (fast, no disk waste)
                    except OSError:
                        try:
                            shutil.copy2(img_path, dst)
                        except OSError:
                            continue
                    collected += 1
                if collected >= max_bg:
                    logger.info("Collected %d background images from datasets", collected)
                    return bg_dir

    logger.info("Collected %d background images from datasets", collected)
    return bg_dir


def _create_solid_backgrounds() -> str:
    """Create simple solid-color background images as fallback."""
    import tempfile

    import cv2
    import numpy as np

    bg_dir = os.path.join(tempfile.gettempdir(), "synthetic_backgrounds")
    os.makedirs(bg_dir, exist_ok=True)

    colors = [
        (100, 120, 80), (80, 100, 120), (120, 80, 100),
        (90, 110, 70), (70, 90, 110), (110, 70, 90),
        (60, 80, 100), (100, 60, 80), (80, 100, 60),
        (50, 70, 90), (90, 50, 70), (70, 90, 50),
    ]

    for i, color in enumerate(colors):
        bg = np.zeros((480, 640, 3), dtype=np.uint8)
        bg[:] = color
        cv2.imwrite(os.path.join(bg_dir, f"solid_bg_{i:04d}.jpg"), bg)

    return bg_dir


def stage_pretrain(
    data_yaml: str,
    base_model: str = "yolo26s.pt",
    output_dir: str = PRETRAIN_DIR,
    epochs: int = 50,
    batch_size: int = 16,
    device: str = "0",
) -> str:
    """Pretrain a strong teacher model on real + synthetic data.

    Returns path to ``best.pt``.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1b: Pretraining teacher model on real + synthetic data")
    logger.info("=" * 60)

    from training.train_yolov26 import train as _train, best_weights_of

    results = _train(
        data=data_yaml,
        model=base_model,
        project=output_dir,
        name="pretrain",
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )

    best = best_weights_of(results)
    if best.exists():
        logger.info("Pretrained model saved: %s", best)
        return str(best)

    logger.warning("best.pt not found after pretraining")
    return ""


def stage_pseudo_label(
    data_yaml: str,
    teacher_weights: str | None = None,
    base_model: str = "yolo26s.pt",
    force_regenerate: bool = False,
) -> int:
    """Run pseudo-labeling with an optional pretrained teacher model.

    Returns the pseudo-labeling exit code.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Pseudo-labeling with teacher model")
    logger.info("=" * 60)

    from pseudo_labeling.__main__ import main as pseudo_main

    args = [
        "pipeline",
        "--data", data_yaml,
        "--merge-mode", "auto",
        "--output-target", "label-set",
        "--non-interactive",
        "--package",
        "--no-train",          # pipeline controls final training separately
    ]
    if force_regenerate:
        args.append("--regenerate")
    else:
        logger.info("Pseudo-labeling will reuse existing label set if available. "
                     "Use --regenerate-pseudo to force regeneration.")

    if teacher_weights and os.path.isfile(teacher_weights):
        args.extend(["--base-model", teacher_weights])
        logger.info("Using pretrained teacher: %s", teacher_weights)
    elif base_model:
        args.extend(["--base-model", base_model])
        logger.info("Using base model: %s", base_model)

    return pseudo_main(args)


def stage_final_train(
    data_yaml: str,
    base_model: str = "yolo26s.pt",
    epochs: int = 100,
    batch_size: int = 16,
    device: str = "0",
    project: str = "angelicam",
    name: str = "yolov26_license_plate",
    use_pseudo_labels: bool = True,
):
    """Train the final model on real + pseudo-labeled data.

    If ``use_pseudo_labels`` is True, resolves the self-supervised label set
    data.yaml and trains on that instead of the raw dataset config.
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: Final training on real + pseudo-labeled data")
    logger.info("=" * 60)

    # Resolve the label-set data.yaml if pseudo-labeling was run
    train_data_yaml = data_yaml
    if use_pseudo_labels:
        resolved = _resolve_label_set_data_yaml(data_yaml)
        if resolved:
            train_data_yaml = resolved
            logger.info("Training on label-set data.yaml: %s", train_data_yaml)
        else:
            logger.warning(
                "Could not resolve label-set data.yaml; falling back to %s", data_yaml
            )

    from training.train_yolov26 import (
        train as _train,
        best_weights_of,
        export_model,
        log_model_to_mlflow,
    )

    results = _train(
        data=train_data_yaml,
        model=base_model,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        project=project,
        name=name,
    )

    best = best_weights_of(results)
    if best.exists():
        exported = export_model(best)
        log_model_to_mlflow(
            best,
            exported,
            params={"commit_hash": _get_git_hash(), "stage": "final"},
        )

    return results


def _resolve_label_set_data_yaml(source_data_yaml: str) -> str | None:
    """Run ``python -m pseudo_labeling select`` to get the label-set data.yaml path."""
    import subprocess
    import json

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pseudo_labeling", "select",
                "--use-label-set", "self_supervised",
                "--data", source_data_yaml,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    # Output is: "emitted data.yaml: /path/to/data.yaml"
    for line in result.stdout.splitlines():
        if "emitted data.yaml:" in line:
            path = line.split("emitted data.yaml:", 1)[-1].strip()
            if os.path.isfile(path):
                return path
    return None


# ── Main ─────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Three-stage training pipeline: pretrain → pseudo-label → final train",
    )

    # Stage control
    p.add_argument("--synthetic", type=int, default=0, metavar="N",
                   help="Generate N synthetic plate images for pretraining (0=skip)")
    p.add_argument("--pretrain", action="store_true",
                   help="Run pretraining stage")
    p.add_argument("--skip-pseudo", dest="skip_pseudo", action="store_true",
                   help="Skip pseudo-labeling stage")
    p.add_argument("--skip-final", dest="skip_final", action="store_true",
                   help="Skip final training stage")

    # Model config
    p.add_argument("--base-model", default="yolo26s.pt",
                   help="Base model for pretraining and/or final training")
    p.add_argument("--teacher-weights",
                   help="Path to pretrained weights for pseudo-labeling teacher")
    p.add_argument("--data", default=DEFAULT_DATA_YAML,
                   help="Dataset YAML path")
    p.add_argument("--device", default="0",
                   help="CUDA device or 'cpu'")

    # Training hyperparams
    p.add_argument("--pretrain-epochs", type=int, default=50,
                   help="Epochs for pretraining stage")
    p.add_argument("--final-epochs", type=int, default=100,
                   help="Epochs for final training stage")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size")
    p.add_argument("--regenerate-synthetic", action="store_true",
                   help="Force regeneration of the synthetic plates dataset")
    p.add_argument("--regenerate-pseudo", action="store_true",
                   help="Force regeneration of pseudo-labels (ignores cached label set)")

    # Output
    p.add_argument("--project", default="angelicam",
                   help="Output project directory")
    p.add_argument("--name", default="yolov26_license_plate",
                   help="Training run name")

    return p


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = build_parser().parse_args(argv)

    commit_hash = _get_git_hash()
    logger.info("Git commit: %s", commit_hash)

    # Keep the original data.yaml path for stages that shouldn't see synthetic data.
    original_data_yaml = args.data
    teacher_weights = args.teacher_weights

    # ── Stage 1: Generate synthetic data + pretrain ──
    if args.synthetic > 0:
        synthetic_yaml = stage_generate_synthetic(
            image_count=args.synthetic,
            seed=args.synthetic,  # deterministic from count
            force=args.regenerate_synthetic,
        )
        # Build a combined data.yaml (real + synthetic) ONLY for pretraining.
        combined_yaml = _inject_synthetic_into_data_yaml(args.data, SYNTHETIC_DATASET_DIR)

        if args.pretrain:
            teacher_weights = stage_pretrain(
                data_yaml=combined_yaml,
                base_model=args.base_model,
                output_dir=os.path.join(args.project, "pretrain"),
                epochs=args.pretrain_epochs,
                batch_size=args.batch_size,
                device=args.device,
            )
    elif args.pretrain:
        # Pretrain on real data only (still produces a strong teacher).
        teacher_weights = stage_pretrain(
            data_yaml=original_data_yaml,
            base_model=args.base_model,
            output_dir=os.path.join(args.project, "pretrain"),
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            device=args.device,
        )

    # ── Stage 2: Pseudo-labeling ──
    # ALWAYS use the original data.yaml here — synthetic plates have no faces,
    # cars or motorcycles, so pseudo-labeling on them is wasteful and noisy.
    if not args.skip_pseudo:
        rc = stage_pseudo_label(
            data_yaml=original_data_yaml,
            teacher_weights=teacher_weights,
            base_model=args.base_model,
            force_regenerate=args.regenerate_pseudo,
        )
        if rc != 0:
            logger.error("Pseudo-labeling failed with exit code %d", rc)
            return rc
    else:
        logger.info("Skipping pseudo-labeling stage")

    # ── Stage 3: Final training ──
    # Resolves the label-set data.yaml (real + pseudo-labels) automatically.
    if not args.skip_final:
        stage_final_train(
            data_yaml=original_data_yaml,
            base_model=args.base_model,
            epochs=args.final_epochs,
            batch_size=args.batch_size,
            device=args.device,
            project=args.project,
            name=args.name,
        )
    else:
        logger.info("Skipping final training stage")

    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())