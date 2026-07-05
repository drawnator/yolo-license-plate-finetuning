"""Pseudo_Labeling_System orchestrator (Req 6-9, 12-14).

Wires the components together into a single ``run(config)`` entry point: scan datasets,
analyze coverage, train class-specialized teachers, generate + confidence-filter
pseudo-labels (resolving RodoSol vehicle classes from folder names), build the audit
report, apply the merge-mode / non-interactive approval gate, merge into the chosen output
target (in-place or a reusable label set), persist a label set + manifest, and return the
manifest.

Designed to run unattended (e.g. inside Docker): with ``merge_mode="auto"`` and
non-interactive execution it completes end-to-end with no prompts.
"""

from __future__ import annotations

import datetime as _dt
import os
import random
import tempfile
from dataclasses import dataclass, field
from typing import Callable

import yaml

from .audit import AuditReport
from .backends import TeacherBackend
from .config import (
    MergeMode,
    OutputTarget,
    PseudoLabelingConfig,
    resolve_merge_mode,
    resolve_non_interactive,
    validate,
)
from .coverage import CoverageAnalyzer, DatasetCoverage, SourceClassMapper
from .generator import NoThresholdConfigured, PseudoLabelGenerator, accepts, resolve_threshold
from .labelset import (
    LABELS_TREE_DIRNAME,
    LabelSet,
    LabelSetArchiveError,
    LabelSetMetadata,
    LabelSetPersistError,
    LabelSetStore,
    make_label_set_id,
    package_label_set,
)
from .manifest import ManifestStoreError, RunManifest, RunManifestStore
from .merger import InPlaceTarget, LabelMerger, LabelSetTarget
from .models import Diagnostic, PseudoLabel
from .scanner import ImageRef, scan_datasets
from .trainer import TeacherTrainer, TeacherTrainingError
from .unified import UNIFIED_IDS, load_unified_space
from .vehicle import VehicleResolutionError, resolve_vehicle_class

#: Approval seam: given the number of proposals, return True to merge (interactive Review_Mode).
ApprovalFn = Callable[[int], bool]

_VEHICLE_CLASSES = frozenset({2, 3})
_RODOSOL_PREFIX = "RodoSol"


@dataclass
class RunOutcome:
    """Result of a full pseudo-labeling run."""

    manifest: RunManifest
    audit: AuditReport
    merged: bool = False
    diagnostics: list[Diagnostic] = field(default_factory=list)


class PseudoLabelingSystem:
    """Top-level orchestrator for a pseudo-labeling run."""

    def __init__(
        self,
        backend: TeacherBackend,
        approval_fn: ApprovalFn | None = None,
        store: LabelSetStore | None = None,
        manifest_store: RunManifestStore | None = None,
        class_mapper: SourceClassMapper | None = None,
    ) -> None:
        self.backend = backend
        self.approval_fn = approval_fn
        self.store = store or LabelSetStore()
        self.manifest_store = manifest_store or RunManifestStore()
        self.coverage_analyzer = CoverageAnalyzer(class_mapper)
        self.trainer = TeacherTrainer(backend)

    def run(self, config: PseudoLabelingConfig) -> RunOutcome:
        """Execute the full pipeline for ``config`` and return the :class:`RunOutcome`."""
        diagnostics: list[Diagnostic] = list(validate(config))
        run_id = config.run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(self.manifest_store.runs_root, run_id)

        # Merge-mode resolution (fatal on ambiguous/unrecognized) -- Req 12.8, 12.9.
        try:
            merge_mode = resolve_merge_mode(config)
        except Exception as exc:  # noqa: BLE001
            return self._fail(run_id, config, diagnostics, str(exc), merge_mode="review")

        non_interactive = resolve_non_interactive(config)

        # Seed setup (Req 9.2, 9.3).
        seed = config.seed if config.seed is not None else random.randrange(2**31)
        random.seed(seed)

        manifest = RunManifest(
            run_id=run_id,
            seed=seed,
            config=_config_summary(config, merge_mode, non_interactive),
            merge_mode=merge_mode.value,
            non_interactive=non_interactive,
            output_target=config.output_target.value,
            thresholds=_thresholds_summary(config),
            status="running",
        )

        # Fatal config diagnostics (e.g. out-of-range threshold) -> stop.
        if any(d.code in {"THRESHOLD_OUT_OF_RANGE",
                          "AMBIGUOUS_MERGE_MODE", "UNRECOGNIZED_MERGE_MODE"} for d in diagnostics):
            manifest.status = "failed"
            self._safe_save(manifest)
            return RunOutcome(manifest=manifest, audit=AuditReport(), diagnostics=diagnostics)

        unified = load_unified_space(config.data_yaml)  # Req 9.4
        scans = scan_datasets(config.data_yaml, config.datasets_root)
        coverages = [
            self.coverage_analyzer.analyze_dataset(s.dataset_id, s.label_files_by_split)
            for s in scans
        ]
        cov_by_id = {c.dataset_id: c for c in coverages}

        generator = PseudoLabelGenerator()
        # accepted[(dataset_id, split, image_path)] -> list[PseudoLabel]
        accepted: dict[tuple[str, str, str], list[PseudoLabel]] = {}
        audit = AuditReport()

        for target_class in sorted(UNIFIED_IDS):
            missing = [s for s in scans if self._is_absent(cov_by_id.get(s.dataset_id), target_class)]
            if not missing:
                continue

            try:
                threshold = resolve_threshold(target_class, config.thresholds)
            except NoThresholdConfigured as exc:
                diagnostics.append(
                    Diagnostic(code="NO_THRESHOLD", message=str(exc), target=str(target_class))
                )
                continue  # Req 4.5 - cannot filter this class; skip it.

            teacher = self._train_teacher(
                target_class, coverages, config, run_dir, seed, manifest, diagnostics
            )
            if teacher is None:
                continue
            model = self.backend.load(teacher)

            for scan in missing:
                for split, refs in scan.images_by_split.items():
                    for ref in refs:
                        self._generate_for_ref(
                            generator, model, ref, target_class, threshold,
                            accepted, audit, diagnostics,
                        )

        diagnostics.extend(generator.diagnostics)

        # Record proposals in the audit report (Req 8.1, 8.5).
        for (dataset_id, _split, _img), labels in accepted.items():
            for pl in labels:
                audit.add_proposal(pl, dataset_id)

        # Zero-accepted-label accounting (Req 7.6).
        for scan in scans:
            for split, refs in scan.images_by_split.items():
                for ref in refs:
                    if not accepted.get((ref.dataset_id, split, ref.image_path)):
                        audit.record_zero_label_image(ref.dataset_id, split)

        # Merge decision (Req 8, 12).
        do_merge = self._should_merge(
            merge_mode, non_interactive, config, audit.total_proposed, diagnostics
        )

        merged = False
        if do_merge and audit.total_proposed > 0:
            merged = self._merge(config, run_id, run_dir, accepted, unified, manifest, audit, diagnostics)

        # Persist audit + manifest (Req 8.1, 9.5).
        os.makedirs(run_dir, exist_ok=True)
        try:
            audit.save(os.path.join(run_dir, "audit_report.json"))
        except OSError as exc:
            diagnostics.append(Diagnostic(code="AUDIT_SAVE_FAILURE", message=str(exc), target=run_dir))

        manifest.status = "success"
        self._safe_save(manifest, diagnostics)
        return RunOutcome(manifest=manifest, audit=audit, merged=merged, diagnostics=diagnostics)

    # ------------------------------------------------------------------
    # generation helpers
    # ------------------------------------------------------------------
    def _generate_for_ref(
        self, generator, model, ref: ImageRef, target_class: int, threshold: float,
        accepted: dict, audit: AuditReport, diagnostics: list[Diagnostic],
    ) -> None:
        candidates = generator.generate_for_image(model, ref.image_path, frozenset({target_class}))
        for cand in candidates:
            if not accepts(cand, threshold):  # Req 4.1
                continue
            class_id = cand.class_id
            # RodoSol vehicle class comes from the folder name (Req 6).
            if ref.dataset_id.startswith(_RODOSOL_PREFIX) and class_id in _VEHICLE_CLASSES:
                try:
                    class_id = resolve_vehicle_class(ref.image_path)
                except VehicleResolutionError as exc:
                    diagnostics.append(
                        Diagnostic(code="VEHICLE_UNRESOLVED", message=str(exc), target=ref.image_path)
                    )
                    continue  # Req 6.4 - exclude vehicle label
            pl = PseudoLabel(
                class_id=class_id, box=cand.box, confidence=cand.confidence,
                image_path=ref.image_path,
            )
            accepted.setdefault((ref.dataset_id, ref.split, ref.image_path), []).append(pl)

    def _train_teacher(
        self, target_class, coverages, config, run_dir, seed, manifest, diagnostics,
    ) -> str | None:
        class_datasets = self.trainer.datasets_containing(target_class, coverages)
        if not class_datasets:
            return None
        teacher_yaml = self._build_teacher_data_yaml(config, class_datasets, run_dir, target_class)
        try:
            result = self.trainer.train_teacher(
                target_class=target_class, coverages=coverages, data_yaml=teacher_yaml,
                run_dir=run_dir, seed=seed, base_model=config.base_model, manifest=manifest,
            )
        except TeacherTrainingError as exc:
            diagnostics.append(
                Diagnostic(code="TEACHER_REJECTED", message=str(exc), target=str(target_class))
            )
            return None
        diagnostics.extend(result.diagnostics)
        return result.weights_path if result.succeeded else None

    def _build_teacher_data_yaml(self, config, class_datasets, run_dir, target_class) -> str:
        """Write a filtered data.yaml containing only class-containing datasets (Req 2.1)."""
        with open(config.data_yaml, "r", encoding="utf-8") as handle:
            src = yaml.safe_load(handle)

        def _filter(entries):
            if isinstance(entries, str):
                entries = [entries]
            return [e for e in entries if _dataset_of(e) in set(class_datasets)]

        out = {
            "path": src.get("path", config.datasets_root),
            "train": _filter(src.get("train", [])),
            "val": _filter(src.get("val", [])),
            "test": _filter(src.get("test", [])),
            "nc": src.get("nc"),
            "names": src.get("names"),
        }
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, f"teacher_class_{target_class}.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(out, handle, sort_keys=False)
        return path

    # ------------------------------------------------------------------
    # merge helpers
    # ------------------------------------------------------------------
    def _should_merge(
        self, merge_mode, non_interactive, config, num_proposed, diagnostics,
    ) -> bool:
        if num_proposed == 0:
            return False  # Req 8.4 / 12.10 - nothing to merge.
        if merge_mode == MergeMode.AUTO_MERGE:
            return True  # Req 12.3
        # Review_Mode.
        if non_interactive:
            if config.assume_yes:
                return True  # Req 12.6 - pre-authorized.
            diagnostics.append(
                Diagnostic(
                    code="REVIEW_DECLINED_NONINTERACTIVE",
                    message="review mode under non-interactive execution without pre-authorized "
                            "approval; proposals treated as declined",
                    target="merge",
                )
            )
            return False  # Req 12.7
        if config.assume_yes:
            return True
        if self.approval_fn is not None:
            return bool(self.approval_fn(num_proposed))
        return False  # default: do not write without explicit approval

    def _merge(
        self, config, run_id, run_dir, accepted, unified, manifest, audit, diagnostics,
    ) -> bool:
        if config.output_target == OutputTarget.LABEL_SET:
            label_set_id = config.label_set_id or make_label_set_id(run_id, config.label_set_name)
            tree_root = os.path.join(run_dir, "label_set")
            labels_tree = os.path.join(tree_root, LABELS_TREE_DIRNAME)
            target = LabelSetTarget(label_set_root=labels_tree, datasets_root=config.datasets_root)
        else:
            label_set_id = None
            target = InPlaceTarget()

        merger = LabelMerger(target)
        for (dataset_id, split, _img), labels in accepted.items():
            image_path = labels[0].image_path
            result = merger.merge_image(
                image_path=image_path, dataset_id=dataset_id, split=split,
                accepted=labels, dup_threshold=config.dup_iou_threshold,
            )
            for d in result.diagnostics:
                audit.write_diagnostics.append(d)
                diagnostics.append(d)

        # Persist the label set (Req 14).
        if config.output_target == OutputTarget.LABEL_SET and label_set_id is not None:
            self._persist_label_set(
                config, run_id, run_dir, label_set_id, manifest, diagnostics
            )
        return True

    def _persist_label_set(self, config, run_id, run_dir, label_set_id, manifest, diagnostics) -> None:
        tree_root = os.path.join(run_dir, "label_set")
        metadata = LabelSetMetadata(
            label_set_id=label_set_id,
            run_id=run_id,
            source_dataset_id=config.datasets_root,
            merge_mode=manifest.merge_mode,
            output_target=manifest.output_target,
            thresholds=manifest.thresholds,
            created_at=_dt.datetime.now().isoformat(timespec="seconds"),
        )
        label_set = LabelSet(label_set_id=label_set_id, root=tree_root, metadata=metadata)
        try:
            location = self.store.persist(label_set)
            manifest.label_set_id = label_set_id
            manifest.label_set_location = location
        except LabelSetPersistError as exc:
            diagnostics.append(
                Diagnostic(code="LABEL_SET_PERSIST_FAILURE", message=str(exc), target=label_set_id)
            )
            manifest.status = "failed"
            return

        if config.package_label_set:
            archive_path = config.label_set_archive_path or os.path.join(
                self.store.root, f"{label_set_id}.tar.gz"
            )
            stored = LabelSet(label_set_id=label_set_id, root=location, metadata=metadata)
            try:
                package_label_set(stored, archive_path)
            except LabelSetArchiveError as exc:
                diagnostics.append(
                    Diagnostic(code="LABEL_SET_ARCHIVE_FAILURE", message=str(exc), target=label_set_id)
                )

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_absent(coverage: DatasetCoverage | None, target_class: int) -> bool:
        if coverage is None:
            return False
        present: set[int] = set()
        for split_cov in coverage.per_split.values():
            present.update(split_cov.present)
        return target_class not in present

    def _fail(self, run_id, config, diagnostics, message, merge_mode) -> RunOutcome:
        diagnostics.append(Diagnostic(code="RUN_FAILED", message=message, target=run_id))
        manifest = RunManifest(run_id=run_id, seed=config.seed or 0, status="failed",
                               merge_mode=merge_mode)
        self._safe_save(manifest)
        return RunOutcome(manifest=manifest, audit=AuditReport(), diagnostics=diagnostics)

    def _safe_save(self, manifest: RunManifest, diagnostics: list[Diagnostic] | None = None) -> None:
        try:
            self.manifest_store.save(manifest)
        except ManifestStoreError as exc:
            if diagnostics is not None:
                diagnostics.append(
                    Diagnostic(code="MANIFEST_STORE_FAILURE", message=str(exc), target=manifest.run_id)
                )


def _dataset_of(entry: str) -> str:
    parts = entry.replace("\\", "/").strip("/").split("/")
    if "images" in parts:
        return "/".join(parts[: parts.index("images")]) or parts[0]
    return parts[0] if parts else entry


def _config_summary(config: PseudoLabelingConfig, merge_mode, non_interactive) -> dict:
    return {
        "merge_mode": merge_mode.value,
        "output_target": config.output_target.value,
        "non_interactive": non_interactive,
        "assume_yes": config.assume_yes,
        "dup_iou_threshold": config.dup_iou_threshold,
        "base_model": config.base_model,
    }


def _thresholds_summary(config: PseudoLabelingConfig) -> dict:
    return {
        "default": config.thresholds.default,
        "per_class": dict(config.thresholds.per_class),
    }
