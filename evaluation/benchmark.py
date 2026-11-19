"""Fixed benchmark: score models the same way so they can be compared over time.

    python -m evaluation.benchmark license-plate-detector@champion runs:/<run_id> path/best.pt

Benchmark ``v1``:

* **Data:** the ``test`` split of every dataset in ``data.yaml``, with the human labels
  of the original datasets. No synthetic images, no pseudo-labels. The data's digest is
  recorded; results are only comparable between runs with the same version and digest.
* **Per dataset, only the classes it annotates.** The datasets are partly labelled (the
  face dataset has no car boxes, RodoSol no faces); scoring every class everywhere would
  count correct detections of unlabelled objects as errors. Each dataset is therefore
  scored on the classes present in its test labels.
* **Fixed settings:** ``imgsz=640``, Ultralytics' validation defaults for confidence and
  IoU, the same for every model.
* **Overall scores** are the mean over datasets (each dataset counts equally, so the
  8,000-image RodoSol set does not drown the 177-image Brazil set). Per-class scores are
  the mean over the datasets that annotate that class.

Results are printed as a table and, when ``MLFLOW_TRACKING_URI`` is set, logged to the
``license-plate-benchmark`` experiment and copied as ``bench.v1.*`` tags onto the
training run and the registered model version. A model already benchmarked with the same
version and data is not re-evaluated unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

from evaluation import model_store

logger = logging.getLogger(__name__)

BENCHMARK_VERSION = "v1"
EXPERIMENT = "license-plate-benchmark"
HEADLINE = ("overall/mAP50-95", "overall/mAP50", "overall/precision", "overall/recall",
            "speed/inference_ms")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def benchmark_sets(data_yaml: str) -> tuple[list[dict], list[str], str]:
    """Return per-dataset test sets, class names and the data digest."""
    from training.tracking import split_dirs, summarize_split

    with open(data_yaml, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    names = cfg["names"] if isinstance(cfg["names"], list) else \
        [cfg["names"][k] for k in sorted(cfg["names"])]
    summary = summarize_split(data_yaml, "test")
    sets = []
    for directory in split_dirs(data_yaml, "test"):
        key = "/".join(directory.parts[-3:])
        counts = summary["sources"].get(key, {})
        classes = sorted(int(c) for c in counts.get("boxes_per_class", {}))
        if counts.get("images") and classes:
            sets.append({"name": _dataset_name(directory), "dir": str(directory),
                         "images": counts["images"], "classes": classes})
    return sets, names, summary["digest"]


def _dataset_name(directory: Path) -> str:
    """``datasets/RodoSol-ALPR/images/testing`` -> ``RodoSol-ALPR``."""
    parts = [p for p in directory.parts if p not in ("images", "test", "testing", "valid")]
    return parts[-1].split(".")[0] if parts else str(directory)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate(weights: Path, sets: list[dict], names: list[str], imgsz: int, batch: int,
             device: str) -> dict[str, float]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics: dict[str, float] = {}
    per_class = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)
    speeds = []
    for ds in sets:
        with tempfile.TemporaryDirectory() as tmp:
            data = {"path": "/", "train": ds["dir"], "val": ds["dir"], "test": ds["dir"],
                    "nc": len(names), "names": names}
            data_file = Path(tmp) / "bench.yaml"
            data_file.write_text(yaml.safe_dump(data))
            result = model.val(data=str(data_file), split="test", imgsz=imgsz, batch=batch,
                               device=device, classes=ds["classes"], plots=False,
                               verbose=False, project=tmp, name="val")
        box = result.box
        values = {"mAP50-95": box.map, "mAP50": box.map50, "precision": box.mp,
                  "recall": box.mr}
        for key, value in values.items():
            metrics[f"dataset/{ds['name']}/{key}"] = float(value)
            overall[key].append(float(value))
        for i, class_id in enumerate(box.ap_class_index):
            p, r, ap50, ap = box.class_result(i)
            cls = names[int(class_id)]
            for key, value in (("mAP50-95", ap), ("mAP50", ap50), ("precision", p),
                               ("recall", r)):
                per_class[cls][key].append(float(value))
        speeds.append((result.speed.get("inference", 0.0), ds["images"]))
    for key, values in overall.items():
        metrics[f"overall/{key}"] = sum(values) / len(values)
    for cls, values in per_class.items():
        for key, vals in values.items():
            metrics[f"class/{cls}/{key}"] = sum(vals) / len(vals)
    total = sum(n for _, n in speeds)
    metrics["speed/inference_ms"] = sum(ms * n for ms, n in speeds) / total if total else 0.0
    return metrics


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
def _tracking() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def _cached(model: model_store.ResolvedModel, digest: str) -> dict[str, float] | None:
    """Metrics of an earlier benchmark of the same model, version and data."""
    import mlflow

    key = _model_key(model)
    if key is None:
        return None
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT], output_format="list", max_results=1,
        filter_string=(f"tags.`benchmark.model` = '{key}' and "
                       f"tags.`benchmark.version` = '{BENCHMARK_VERSION}' and "
                       f"tags.`benchmark.data_digest` = '{digest}'"),
        order_by=["attributes.start_time DESC"])
    return dict(runs[0].data.metrics) if runs else None


def _model_key(model: model_store.ResolvedModel) -> str | None:
    """Stable identity of a model across benchmark runs (local files have none)."""
    if model.registry:
        return f"models:/{model.registry[0]}/{model.registry[1]}"
    if model.run_id:
        return f"runs:/{model.run_id}"
    return None


def _log(model: model_store.ResolvedModel, metrics: dict, digest: str, sets: list[dict],
         settings: dict) -> None:
    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient()
    mlflow.set_experiment(EXPERIMENT)
    key = _model_key(model) or str(model.weights)
    with mlflow.start_run(run_name=f"{BENCHMARK_VERSION} {key}"):
        mlflow.set_tags({"benchmark.version": BENCHMARK_VERSION,
                         "benchmark.data_digest": digest, "benchmark.model": key,
                         **({"benchmark.training_run": model.run_id} if model.run_id else {})})
        mlflow.log_params(settings)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"datasets": sets, "metrics": metrics}, "benchmark.json")
    # Copy the headline numbers onto the model itself, so lists of runs and registry
    # versions show them without opening the benchmark experiment.
    # e.g. bench.v1.mAP50-95, bench.v1.speed.inference_ms
    headline = {}
    for key in HEADLINE:
        if key in metrics:
            short = key.split("/", 1)[1] if key.startswith("overall/") else key.replace("/", ".")
            headline[f"bench.{BENCHMARK_VERSION}.{short}"] = f"{metrics[key]:.4f}"
    headline[f"bench.{BENCHMARK_VERSION}.data_digest"] = digest
    for tag, value in headline.items():
        if model.run_id:
            client.set_tag(model.run_id, tag, value)
        if model.registry:
            client.set_model_version_tag(model.registry[0], model.registry[1], tag, value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_table(results: list[tuple[str, dict]]) -> None:
    keys = list(HEADLINE) + sorted({k for _, m in results for k in m} - set(HEADLINE))
    width = max(len(k) for k in keys)
    cols = [ref if len(ref) <= 28 else "..." + ref[-25:] for ref, _ in results]
    print(f"{'benchmark ' + BENCHMARK_VERSION:<{width}}  " + "  ".join(f"{c:>28}" for c in cols))
    for key in keys:
        row = []
        for _, m in results:
            v = m.get(key)
            row.append(f"{v:>28.4f}" if isinstance(v, (int, float)) else f"{'-':>28}")
        print(f"{key:<{width}}  " + "  ".join(row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.benchmark",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="+", help="file, runs:/<id>, models:/<name>@<alias> ...")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "models"))
    parser.add_argument("--force", action="store_true", help="re-evaluate cached results")
    parser.add_argument("--no-log", action="store_true", help="do not write to MLflow")
    parser.add_argument("--json", help="also write all results to this file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")

    sets, names, digest = benchmark_sets(args.data)
    logger.info("Benchmark %s, data digest %s: %s", BENCHMARK_VERSION, digest,
                ", ".join(f"{s['name']} ({s['images']} images, classes "
                          f"{[names[c] for c in s['classes']]})" for s in sets))
    settings = {"imgsz": args.imgsz, "batch": args.batch, "data": args.data}
    results = []
    for ref in args.models:
        model = model_store.resolve(ref, args.models_dir)
        metrics = None
        if _tracking() and not args.force:
            metrics = _cached(model, digest)
            if metrics:
                logger.info("%s: using the earlier %s result (--force to redo)", ref,
                            BENCHMARK_VERSION)
        if metrics is None:
            logger.info("%s: evaluating %s", ref, model.weights)
            metrics = evaluate(model.weights, sets, names, args.imgsz, args.batch, args.device)
            if _tracking() and not args.no_log:
                _log(model, metrics, digest, sets, settings)
        results.append((ref, metrics))
    print_table(results)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
