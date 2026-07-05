"""Label_Set persistence and packaging (Req 13, 14).

A :class:`LabelSet` is a complete, self-contained label directory tree (a copy of the
originals plus merged pseudo-labels). The :class:`LabelSetStore` persists it under a stable
id so it can be reused in future training runs without regenerating, and
:func:`package_label_set` bundles it (plus metadata) into a distributable archive that can
be downloaded alongside the dataset.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
from dataclasses import asdict, dataclass, field

#: Filename of the metadata document stored inside a label set (Req 14.2).
METADATA_FILENAME = "metadata.json"
#: Sub-directory (under the label set id) holding the mirrored label tree.
LABELS_TREE_DIRNAME = "labels-tree"


class LabelSetPersistError(Exception):
    """Raised when a Label_Set cannot be persisted to the store (Req 14.6)."""


class LabelSetArchiveError(Exception):
    """Raised when a Label_Set_Archive cannot be produced (Req 14.7)."""


class LabelSetNotFound(Exception):
    """Raised when a Label_Set_Id does not resolve to a stored label set (Req 15.6)."""


@dataclass(frozen=True)
class LabelSetMetadata:
    """Traceability record persisted alongside a label set (Req 14.2)."""

    label_set_id: str
    run_id: str
    source_dataset_id: str
    merge_mode: str
    output_target: str
    thresholds: dict = field(default_factory=dict)
    created_at: str = ""


@dataclass(frozen=True)
class LabelSet:
    """A complete label directory tree separate from the originals (Req 13)."""

    label_set_id: str
    root: str  # directory containing labels-tree/ and metadata.json
    metadata: LabelSetMetadata


def _slugify(name: str) -> str:
    """Return a filesystem-safe slug for a user-supplied label-set name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-")
    return slug or "labelset"


def make_label_set_id(run_id: str, name: str | None = None) -> str:
    """Derive a stable, filesystem-safe Label_Set_Id (Req 14.1).

    Uses a slugified operator-supplied ``name`` when given, otherwise derives one from the
    producing ``run_id`` as ``ls-{run_id}``.
    """
    if name:
        return _slugify(name)
    return f"ls-{run_id}"


class LabelSetStore:
    """Persisted, named location for label sets, retrievable by Label_Set_Id (Req 14.1)."""

    def __init__(self, root: str = "datasets/label_sets") -> None:
        self.root = root

    def location_for(self, label_set_id: str) -> str:
        """Return the stable store path for a label set id (Req 14.1)."""
        return os.path.join(self.root, label_set_id)

    def labels_tree_dir(self, label_set_id: str) -> str:
        """Return the directory that should hold the mirrored label tree for an id."""
        return os.path.join(self.location_for(label_set_id), LABELS_TREE_DIRNAME)

    def persist(self, label_set: LabelSet) -> str:
        """Persist a label set (tree + metadata) to its stable location (Req 14.1, 14.5).

        Idempotent: persisting the same id regenerates the same location, so repeated runs
        with identical inputs produce an equivalent stored label set.

        Raises:
            LabelSetPersistError: if the label set cannot be written (Req 14.6).
        """
        dest = self.location_for(label_set.label_set_id)
        try:
            os.makedirs(dest, exist_ok=True)
            # Move/copy the produced tree into the store location under labels-tree/.
            dest_tree = os.path.join(dest, LABELS_TREE_DIRNAME)
            src_tree = os.path.join(label_set.root, LABELS_TREE_DIRNAME)
            if os.path.abspath(src_tree) != os.path.abspath(dest_tree) and os.path.isdir(src_tree):
                if os.path.isdir(dest_tree):
                    shutil.rmtree(dest_tree)
                shutil.copytree(src_tree, dest_tree)
            os.makedirs(dest_tree, exist_ok=True)
            with open(os.path.join(dest, METADATA_FILENAME), "w", encoding="utf-8") as handle:
                json.dump(asdict(label_set.metadata), handle, indent=2, sort_keys=True)
            return dest
        except OSError as exc:
            raise LabelSetPersistError(
                f"could not persist label set {label_set.label_set_id!r} to {dest!r}: {exc}"
            ) from exc

    def load(self, label_set_id: str) -> LabelSet:
        """Retrieve a persisted label set by id (Req 14.1).

        Raises:
            LabelSetNotFound: if the id does not resolve to a stored label set (Req 15.6).
        """
        dest = self.location_for(label_set_id)
        meta_path = os.path.join(dest, METADATA_FILENAME)
        if not os.path.isfile(meta_path):
            raise LabelSetNotFound(f"no label set found for id {label_set_id!r} at {dest!r}")
        with open(meta_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return LabelSet(
            label_set_id=label_set_id,
            root=dest,
            metadata=LabelSetMetadata(**data),
        )


def package_label_set(label_set: LabelSet, archive_path: str) -> str:
    """Bundle a label set (tree + metadata) into a .tar.gz archive (Req 14.4).

    Raises:
        LabelSetArchiveError: on failure; the persisted label set is left unchanged
            (Req 14.7).
    """
    try:
        parent = os.path.dirname(archive_path) or "."
        os.makedirs(parent, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(label_set.root, arcname=label_set.label_set_id)
        return archive_path
    except (OSError, tarfile.TarError) as exc:
        raise LabelSetArchiveError(
            f"could not package label set {label_set.label_set_id!r} to {archive_path!r}: {exc}"
        ) from exc
