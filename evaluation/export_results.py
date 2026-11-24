"""Collect benchmark results from MLflow into one CSV for the dashboard.

    python -m evaluation.export_results            # -> evaluation/results/benchmark_v1.csv

One row per benchmarked model (latest result per weights file), joined with its entry in
``evaluation/models.yaml`` and, where known, its MLflow training run and Model Registry
version. Models benchmarked outside the catalog (for example a new ``@candidate``) are
included too, named after their reference.

Columns: descriptive fields first (``id``, ``size``, ``date``, ``pipeline``, ``notes``,
links, ``aliases``), then ``train_val_mAP50-95`` (the score the training itself reported;
not comparable between models, shown for reference), then every benchmark metric
(``overall/…``, ``class/…``, ``dataset/…``, ``dsclass/<dataset>/<class>/<metric>``,
``speed/inference_ms``).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from evaluation import benchmark, model_store

INFO_COLUMNS = ["id", "ref", "size", "date", "pipeline", "notes", "partial", "classes", "aliases",
                "registry_version", "training_run", "mlflow_url", "release_url",
                "weights_sha256", "benchmark_run", "train_val_mAP50-95"]


def collect(catalog_path: str, models_dir: str) -> list[dict]:
    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient()
    tracking = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")
    catalog = {m["id"]: m for m in benchmark.load_catalog(catalog_path)}
    runs = mlflow.search_runs(
        experiment_names=[benchmark.EXPERIMENT], output_format="list", max_results=5000,
        filter_string=f"tags.`benchmark.version` = '{benchmark.BENCHMARK_VERSION}'",
        order_by=["attributes.start_time DESC"])

    # Only results on the current benchmark data, newest per weights file.
    digest = runs[0].data.tags.get("benchmark.data_digest") if runs else None
    latest: dict[str, object] = {}
    for run in runs:
        tags = run.data.tags
        sha = tags.get("benchmark.weights_sha256")
        if not sha or tags.get("benchmark.data_digest") != digest or sha in latest:
            continue
        latest[sha] = run

    aliases, versions = _registry(client)
    rows = []
    for sha, run in latest.items():
        tags = run.data.tags
        entry = catalog.get(tags.get("benchmark.catalog_id", ""), {})
        model_key = tags.get("benchmark.model", "")
        training_run = tags.get("benchmark.training_run") or entry.get("run_id") or ""
        registry = versions.get(training_run)
        row = {
            "id": entry.get("id") or model_key,
            "ref": entry.get("ref") or model_key,
            "size": entry.get("size", ""),
            "date": str(entry.get("date", "")),
            "pipeline": entry.get("pipeline", ""),
            "notes": entry.get("notes", ""),
            "partial": "true" if entry.get("partial") else "",
            "aliases": " ".join(sorted(aliases.get(registry, []))) if registry else "",
            "registry_version": registry or "",
            "training_run": training_run,
            "weights_sha256": sha,
            "benchmark_run": run.info.run_id,
            "release_url": _release_url(entry.get("ref", "")),
        }
        if training_run:
            try:
                train = client.get_run(training_run)
            except Exception:  # noqa: BLE001 - a deleted run should not stop the export
                train = None
            if train is not None:
                exp = train.info.experiment_id
                row["mlflow_url"] = f"{tracking}/#/experiments/{exp}/runs/{training_run}"
                value = train.data.metrics.get("metrics/mAP50-95B")
                if value is not None:
                    row["train_val_mAP50-95"] = round(value, 4)
                if not row["date"]:
                    import datetime as dt
                    row["date"] = dt.date.fromtimestamp(train.info.start_time / 1000).isoformat()
        if "train_val_mAP50-95" not in row:
            row["train_val_mAP50-95"] = _checkpoint_score(entry.get("ref", ""), models_dir)
        metrics = dict(run.data.metrics)
        row["classes"] = " ".join(sorted({k.split("/")[1] for k in metrics
                                          if k.startswith("class/")}))
        row.update({k: round(v, 5) for k, v in sorted(metrics.items())})
        rows.append(row)

    order = list(catalog)
    rows.sort(key=lambda r: (order.index(r["id"]) if r["id"] in order else len(order),
                             r["date"], r["id"]))
    return rows


def _registry(client) -> tuple[dict, dict]:
    """``{version: [aliases]}`` and ``{training run id: version}`` for the registered model."""
    name = os.environ.get("MLFLOW_REGISTERED_MODEL", "license-plate-detector")
    try:
        model = client.get_registered_model(name)
        found = client.search_model_versions(f"name = '{name}'")
    except Exception:  # noqa: BLE001 - no registry yet
        return {}, {}
    aliases: dict[str, list[str]] = {}
    for alias, version in (model.aliases or {}).items():
        aliases.setdefault(str(version), []).append(alias)
    return aliases, {v.run_id: str(v.version) for v in found}


def _release_url(ref: str) -> str:
    if not ref.startswith("release:"):
        return ""
    return f"https://github.com/{model_store.GITHUB_REPO}/releases/tag/{ref[len('release:'):]}"


def _checkpoint_score(ref: str, models_dir: str):
    """Validation score stored in a release checkpoint, if it was downloaded."""
    if not ref.startswith("release:"):
        return ""
    files = sorted((Path(models_dir) / "releases" / ref[len("release:"):]).glob("*.pt"))
    if not files:
        return ""
    try:
        import torch

        ckpt = torch.load(files[0], map_location="cpu", weights_only=False)
        value = (ckpt.get("train_metrics") or {}).get("metrics/mAP50-95(B)")
        return round(float(value), 4) if value is not None else ""
    except Exception:  # noqa: BLE001
        return ""


def write_csv(rows: list[dict], path: Path) -> None:
    metric_cols = sorted({k for r in rows for k in r} - set(INFO_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INFO_COLUMNS + metric_cols)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.export_results",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--catalog", default="evaluation/models.yaml")
    parser.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "models"))
    parser.add_argument("--out", default=f"evaluation/results/benchmark_"
                                         f"{benchmark.BENCHMARK_VERSION}.csv")
    args = parser.parse_args(argv)
    rows = collect(args.catalog, args.models_dir)
    write_csv(rows, Path(args.out))
    print(f"{len(rows)} models -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
