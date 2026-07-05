"""CLI for the pseudo-labeling pipeline (Req 8, 12, 13, 14, 15).

Subcommands
-----------
``run``       Run pseudo-labeling only (coverage -> teachers -> generate -> merge).
``select``    Emit a ``data.yaml`` pointing training at a saved label set (or the originals).
``pipeline``  One-shot, unattended: self-supervise (auto-merge, non-interactive) then run
              normal training via ``training/train_yolov26.py`` with no manual input.
              Designed for the Docker training container.

Example (Docker, fully unattended):

    python -m pseudo_labeling pipeline --output-target label-set --package

which pseudo-labels every dataset, saves a reusable label set + archive, then trains on it.
"""

from __future__ import annotations

import argparse
import sys

import yaml

from .config import (
    MergeMode,
    OutputTarget,
    PseudoLabelingConfig,
    ThresholdConfig,
)
from .dataset_config import ORIGINAL_LABELS, LabelSetSelector
from .labelset import LabelSetStore
from .system import PseudoLabelingSystem


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------
def _load_config(args: argparse.Namespace) -> PseudoLabelingConfig:
    """Merge a --config YAML (if any) with CLI flags into a PseudoLabelingConfig."""
    file_cfg: dict = {}
    if getattr(args, "config", None):
        with open(args.config, "r", encoding="utf-8") as handle:
            file_cfg = yaml.safe_load(handle) or {}

    thresholds = ThresholdConfig(
        default=_first_not_none(getattr(args, "threshold", None), file_cfg.get("threshold"), 0.5),
        per_class=_parse_class_thresholds(getattr(args, "class_threshold", None)) or file_cfg.get("per_class", {}),
    )

    merge_raw = getattr(args, "merge_mode", None) or file_cfg.get("merge_mode")

    return PseudoLabelingConfig(
        data_yaml=_first_not_none(getattr(args, "data", None), file_cfg.get("data_yaml"), "./data.yaml"),
        datasets_root=_first_not_none(getattr(args, "datasets_root", None), file_cfg.get("datasets_root"), "./datasets"),
        base_model=_first_not_none(getattr(args, "base_model", None), file_cfg.get("base_model")),
        thresholds=thresholds,
        dup_iou_threshold=_first_not_none(getattr(args, "dup_iou", None), file_cfg.get("dup_iou_threshold"), 0.5),
        merge_mode=MergeMode(merge_raw) if merge_raw in ("review", "auto") else MergeMode.REVIEW,
        merge_mode_raw=merge_raw,
        non_interactive=getattr(args, "non_interactive", None),
        assume_yes=bool(getattr(args, "assume_yes", False)),
        output_target=OutputTarget(getattr(args, "output_target", None)) if getattr(args, "output_target", None) else OutputTarget.IN_PLACE,
        label_set_id=getattr(args, "label_set_id", None),
        label_set_name=getattr(args, "label_set_name", None),
        package_label_set=bool(getattr(args, "package", False)),
        seed=getattr(args, "seed", None),
        run_id=getattr(args, "run_id", None),
    )


def _first_not_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def _parse_class_thresholds(pairs: list[str] | None) -> dict[int, float]:
    """Parse ``--class-threshold plate=0.6`` style pairs into {class_id: threshold}."""
    if not pairs:
        return {}
    names = {"plate": 0, "face": 1, "car": 2, "motorcycle": 3}
    out: dict[int, float] = {}
    for pair in pairs:
        key, _, val = pair.partition("=")
        key = key.strip()
        class_id = names.get(key, None)
        if class_id is None and key.isdigit():
            class_id = int(key)
        if class_id is not None:
            out[class_id] = float(val)
    return out


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    system = PseudoLabelingSystem(backend=_make_backend(args), approval_fn=_interactive_approval)
    outcome = system.run(cfg)
    _print_outcome(outcome)
    return 0 if outcome.manifest.status == "success" else 1


def _cmd_select(args: argparse.Namespace) -> int:
    store = LabelSetStore(root=getattr(args, "store_root", None) or "datasets/label_sets")
    selector = LabelSetSelector(store=store, source_data_yaml=args.data or "./data.yaml")
    selection = args.use_label_set
    try:
        cfg = selector.select(selection, output_root=args.output_root or "runs/label_set_selection")
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"emitted data.yaml: {cfg.yaml_path}")
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    """Unattended self-supervise -> train (Docker one-shot)."""
    # Default the self-supervision step to unattended, non-destructive, reusable output.
    if getattr(args, "merge_mode", None) is None:
        args.merge_mode = "auto"
    if getattr(args, "output_target", None) is None:
        args.output_target = "label-set"
    if getattr(args, "non_interactive", None) is None:
        args.non_interactive = True

    cfg = _load_config(args)
    system = PseudoLabelingSystem(backend=_make_backend(args))
    outcome = system.run(cfg)
    _print_outcome(outcome)
    if outcome.manifest.status != "success":
        print("pseudo-labeling failed; aborting training", file=sys.stderr)
        return 1

    # Resolve the data.yaml to train on: the produced label set if any, else the original.
    train_data = cfg.data_yaml
    if outcome.manifest.label_set_id:
        store = LabelSetStore(root=cfg.label_set_store_root)
        selector = LabelSetSelector(store=store, source_data_yaml=cfg.data_yaml)
        dscfg = selector.select(
            outcome.manifest.label_set_id,
            output_root=f"runs/pseudo_labeling/{outcome.manifest.run_id}/train_data",
        )
        train_data = dscfg.yaml_path
        print(f"training will use label-set data.yaml: {train_data}")

    if args.no_train:
        print("`--no-train` set; skipping training step")
        return 0

    from training.train_yolov26 import train as _train

    _train(data=train_data, model=cfg.base_model or "yolo26s.pt")
    return 0


def _make_backend(args: argparse.Namespace):
    """Return the teacher backend (real Ultralytics)."""
    from .backends import UltralyticsBackend

    return UltralyticsBackend()


def _interactive_approval(num_proposed: int) -> bool:
    reply = input(f"Merge {num_proposed} proposed pseudo-labels? [y/N] ").strip().lower()
    return reply in ("y", "yes")


def _print_outcome(outcome) -> None:
    m = outcome.manifest
    print(f"run {m.run_id}: status={m.status} merge_mode={m.merge_mode} "
          f"output_target={m.output_target} proposed={outcome.audit.total_proposed} "
          f"merged={outcome.merged}")
    if m.label_set_id:
        print(f"label set: {m.label_set_id} @ {m.label_set_location}")
    for d in outcome.diagnostics[:20]:
        print(f"  [{d.code}] {d.message}")


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------
def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config")
    p.add_argument("--data", dest="data")
    p.add_argument("--datasets-root", dest="datasets_root")
    p.add_argument("--base-model", dest="base_model")
    p.add_argument("--threshold", type=float)
    p.add_argument("--class-threshold", dest="class_threshold", action="append")
    p.add_argument("--dup-iou", dest="dup_iou", type=float)
    p.add_argument("--merge-mode", dest="merge_mode", choices=["review", "auto"])
    p.add_argument("--yes", "--assume-yes", dest="assume_yes", action="store_true")
    p.add_argument("--non-interactive", dest="non_interactive", action="store_true", default=None)
    p.add_argument("--output-target", dest="output_target", choices=["in-place", "label-set"])
    p.add_argument("--label-set-id", dest="label_set_id")
    p.add_argument("--label-set-name", dest="label_set_name")
    p.add_argument("--package", "--archive", dest="package", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--run-id", dest="run_id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pseudo_labeling")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run pseudo-labeling only")
    _add_common_flags(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_sel = sub.add_parser("select", help="emit a data.yaml for a saved label set or the originals")
    p_sel.add_argument("--use-label-set", dest="use_label_set", default=ORIGINAL_LABELS,
                       help="label-set id, or 'original'")
    p_sel.add_argument("--data", dest="data")
    p_sel.add_argument("--store-root", dest="store_root")
    p_sel.add_argument("--output-root", dest="output_root")
    p_sel.set_defaults(func=_cmd_select)

    p_pipe = sub.add_parser("pipeline", help="unattended: self-supervise then train (Docker)")
    _add_common_flags(p_pipe)
    p_pipe.add_argument("--no-train", dest="no_train", action="store_true",
                        help="run pseudo-labeling + label-set selection but skip training")
    p_pipe.set_defaults(func=_cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
