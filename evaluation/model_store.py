"""Find a trained model's weights from a file path, an MLflow run or the Model Registry.

Accepted references:

``path/to/best.pt``                       a local file
``release:<tag>``                         the ``.pt`` asset of a GitHub release of this repo
``runs:/<run_id>``                        ``weights/best.pt`` of that MLflow run
``models:/<name>/<version>``              a registered model version
``models:/<name>@<alias>``                a registered model alias (``champion``, ...)
``<name>@<alias>`` / ``<name>/<version>``  same, without the ``models:/`` prefix

MLflow references need ``MLFLOW_TRACKING_URI`` (and credentials) in the environment.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
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
    if ref.startswith("release:"):
        return ResolvedModel(ref=ref, weights=_download_release(ref[len("release:"):],
                                                                Path(out_dir) / "releases"))

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


GITHUB_REPO = os.environ.get("GITHUB_REPO", "drawnator/yolo-license-plate-finetuning")


def _download_release(tag: str, out_dir: Path) -> Path:
    """Download the single ``.pt`` asset of release ``tag`` (cached in ``out_dir/<tag>``)."""
    target = out_dir / tag
    cached = sorted(target.glob("*.pt")) if target.is_dir() else []
    if cached:
        return cached[0]
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{tag}"
    request = urllib.request.Request(url, headers={"User-Agent": "model-store",
                                                   "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        assets = [a for a in json.load(response)["assets"] if a["name"].endswith(".pt")]
    if len(assets) != 1:
        raise ValueError(f"release {tag!r} has {len(assets)} .pt assets, expected 1")
    target.mkdir(parents=True, exist_ok=True)
    path = target / assets[0]["name"]
    partial = path.with_suffix(".part")
    urllib.request.urlretrieve(assets[0]["browser_download_url"], partial)
    partial.rename(path)
    return path


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
