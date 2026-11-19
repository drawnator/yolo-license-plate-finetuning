"""Find a trained model's weights from a file path, an MLflow run or the Model Registry.

Accepted references:

``path/to/best.pt``                       a local file
``runs:/<run_id>``                        ``weights/best.pt`` of that MLflow run
``models:/<name>/<version>``              a registered model version
``models:/<name>@<alias>``                a registered model alias (``champion``, ...)
``<name>@<alias>`` / ``<name>/<version>``  same, without the ``models:/`` prefix

MLflow references need ``MLFLOW_TRACKING_URI`` (and credentials) in the environment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ResolvedModel:
    ref: str                      # what the user passed
    weights: Path                 # local best.pt
    run_id: str | None = None     # MLflow training run, when known
    registry: tuple[str, str] | None = None  # (name, version), when registered
    info: dict = field(default_factory=dict)


def resolve(ref: str, out_dir: str | Path = "models") -> ResolvedModel:
    """Return the local weights for ``ref``, downloading them if needed."""
    if Path(ref).is_file():
        return ResolvedModel(ref=ref, weights=Path(ref))

    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient()
    name_ref = ref[len("models:/"):] if ref.startswith("models:/") else ref

    if ref.startswith("runs:/"):
        run_id = ref[len("runs:/"):].split("/")[0]
        return ResolvedModel(ref=ref, weights=_download(f"runs:/{run_id}/weights/best.pt",
                                                        Path(out_dir) / f"run-{run_id[:8]}"),
                             run_id=run_id)

    match = re.fullmatch(r"([^@/]+)(?:@([\w-]+)|/(\d+))", name_ref)
    if not match:
        raise ValueError(f"not a file, runs:/ or registry reference: {ref!r}")
    name, alias, number = match.groups()
    version = (client.get_model_version_by_alias(name, alias) if alias
               else client.get_model_version(name, number))
    target = Path(out_dir) / f"{name}-v{version.version}"
    weights = _download(version.source.rstrip("/") + "/best.pt", target)
    return ResolvedModel(ref=ref, weights=weights, run_id=version.run_id,
                         registry=(name, version.version),
                         info={"description": version.description or "",
                               "tags": dict(version.tags or {})})


def _download(artifact_uri: str, target: Path) -> Path:
    import mlflow

    existing = target / "best.pt"
    if existing.is_file():  # versions never change, so a downloaded copy can be reused
        return existing
    target.mkdir(parents=True, exist_ok=True)
    path = Path(mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri,
                                                    dst_path=str(target)))
    return path if path.is_file() else path / "best.pt"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m evaluation.model_store",
        description="Download a model's best.pt (see the module docstring for references).")
    parser.add_argument("ref", help="e.g. license-plate-detector@champion")
    parser.add_argument("--out", default=os.environ.get("MODELS_DIR", "models"))
    args = parser.parse_args(argv)
    model = resolve(args.ref, args.out)
    print(model.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
