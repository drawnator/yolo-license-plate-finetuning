"""Label_Set_Selector: emit a data.yaml-compatible config for training (Req 15).

At training time an operator selects either the original labels or a saved label set. For
a saved label set this module materializes a *generated dataset root* whose ``images``
directories are symlinks/hardlinks back to the original images and whose sibling ``labels``
directories are the selected label set -- so Ultralytics' fixed ``images`` -> ``labels``
path derivation lands on the label-set labels while the image bytes stay the originals and
every original file/dir is left untouched.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

import yaml

from .labelset import LABELS_TREE_DIRNAME, LabelSetNotFound, LabelSetStore
from .unified import load_unified_space

#: Sentinel selecting the untouched Original_Labels rather than a saved label set (Req 15.1).
ORIGINAL_LABELS = "original"


@dataclass
class DatasetConfig:
    """A data.yaml-compatible configuration emitted for a training run (Req 15.2)."""

    path: str
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    nc: int = 0
    names: list[str] = field(default_factory=list)
    yaml_path: str = ""

    def to_yaml_dict(self) -> dict:
        return {
            "path": self.path,
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "nc": self.nc,
            "names": self.names,
        }


class LabelSetSelector:
    """Produce a data.yaml pointing training at the original labels or a saved label set."""

    def __init__(self, store: LabelSetStore, source_data_yaml: str = "./data.yaml") -> None:
        self.store = store
        self.source_data_yaml = source_data_yaml

    def select(self, selection: str, output_root: str) -> DatasetConfig:
        """Select the label source and emit a ready-to-use ``data.yaml`` (Req 15).

        ``selection`` is either :data:`ORIGINAL_LABELS` or a Label_Set_Id.

        Raises:
            LabelSetNotFound: if a label-set id does not resolve in the store (Req 15.6).
        """
        source = self._read_source()
        names = source["names"]
        nc = source.get("nc", len(names))

        if selection == ORIGINAL_LABELS:
            # Point training straight at the existing dataset root; originals untouched.
            cfg = DatasetConfig(
                path=source.get("path", "./datasets"),
                train=source.get("train", []),
                val=source.get("val", []),
                test=source.get("test", []),
                nc=nc,
                names=names,
            )
            return self._emit(cfg, os.path.join(output_root, "data.yaml"))

        # A saved label set: verify it exists (raises LabelSetNotFound -> Req 15.6).
        label_set = self.store.load(selection)
        generated_root = os.path.join(output_root, selection)
        self._materialize_generated_root(source, label_set.root, generated_root)

        cfg = DatasetConfig(
            path=generated_root,
            train=source.get("train", []),
            val=source.get("val", []),
            test=source.get("test", []),
            nc=nc,
            names=names,
        )
        return self._emit(cfg, os.path.join(generated_root, "data.yaml"))

    # ------------------------------------------------------------------
    def _read_source(self) -> dict:
        with open(self.source_data_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        # Validate class space is loadable (raises on nc/names mismatch).
        load_unified_space(self.source_data_yaml)
        return data

    def _materialize_generated_root(
        self, source: dict, label_set_root: str, generated_root: str
    ) -> None:
        """Link original images and place label-set labels under a generated root (Req 15).

        Images are linked (symlink, falling back to copy) so their bytes remain the
        originals; labels come from the label set tree. Original dirs are never modified.
        """
        os.makedirs(generated_root, exist_ok=True)
        datasets_root = source.get("path", "./datasets")
        labels_tree = os.path.join(label_set_root, LABELS_TREE_DIRNAME)

        # Collect the image dirs referenced by the source data.yaml across splits.
        image_dirs: list[str] = []
        for key in ("train", "val", "test"):
            entries = source.get(key, [])
            if isinstance(entries, str):
                entries = [entries]
            image_dirs.extend(entries)

        for rel_img_dir in image_dirs:
            src_img_dir = os.path.join(datasets_root, rel_img_dir)
            dst_img_dir = os.path.join(generated_root, rel_img_dir)
            self._link_tree(src_img_dir, dst_img_dir)

            # The sibling labels dir for this split comes from the label set tree.
            rel_lbl_dir = _swap_images_for_labels(rel_img_dir)
            src_lbl_dir = os.path.join(labels_tree, rel_lbl_dir)
            dst_lbl_dir = os.path.join(generated_root, rel_lbl_dir)
            if os.path.isdir(src_lbl_dir):
                self._link_tree(src_lbl_dir, dst_lbl_dir)

    @staticmethod
    def _link_tree(src: str, dst: str) -> None:
        """Link ``src`` to ``dst`` (symlink; fall back to copytree if unsupported)."""
        if not os.path.exists(src):
            return
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        if os.path.lexists(dst):
            return
        try:
            os.symlink(os.path.abspath(src), dst, target_is_directory=True)
        except (OSError, NotImplementedError):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    def _emit(self, cfg: DatasetConfig, yaml_path: str) -> DatasetConfig:
        os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg.to_yaml_dict(), handle, sort_keys=False)
        cfg.yaml_path = yaml_path
        return cfg


def _swap_images_for_labels(rel_dir: str) -> str:
    """Swap the last ``images`` path segment for ``labels`` in a relative dir path."""
    parts = rel_dir.replace("\\", "/").split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return "/".join(parts)
