"""Coverage_Analyzer: per-dataset / per-split unified-class coverage (Req 1).

The analyzer reads a dataset's YOLO label files and determines, **separately for each
Dataset_Split**, which unified class ids are *present* (have at least one valid
annotation resolving to them) and which are *absent* (the explicit complement over the
full Unified_Class_Space ``{0, 1, 2, 3}``).

Source annotations are first mapped to unified ids by an injectable
:class:`SourceClassMapper` (Req 1.3) before presence is computed, so datasets that store
annotations in a non-unified encoding (e.g. UFPR-ALPR line-2 vehicle codes, or any
dataset-local class ids) can be normalised without touching the analysis logic.

Robustness (Req 1.5-1.7):

* A label file that cannot be **opened** (``OSError``) is treated as unreadable; when a
  dataset's label files are all unopenable the :class:`DatasetCoverage` is flagged
  ``unreadable`` with an empty present set and a diagnostic identifying the dataset
  (Req 1.5).
* A label file that opens but contains **malformed** annotation data (a bad class id or
  geometry field) is excluded in its entirety; analysis continues with the remaining
  files and a diagnostic identifies the excluded file (Req 1.6).
* A single annotation whose source class has **no unified mapping** or whose id falls
  outside ``0..3`` is excluded from presence determination while the rest of its file is
  kept, and a diagnostic identifies the offending class id (Req 1.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Diagnostic
from .unified import UNIFIED_IDS
from .yolo_format import parse_line

#: Diagnostic code for a label file that could not be opened (Req 1.5/1.6 open failure).
UNREADABLE_LABEL = "UNREADABLE_LABEL"
#: Diagnostic code for a dataset whose label files were all unopenable (Req 1.5).
UNREADABLE_DATASET = "UNREADABLE_DATASET"
#: Diagnostic code for a label file excluded because it contained malformed data (Req 1.6).
MALFORMED_LABEL = "MALFORMED_LABEL"
#: Diagnostic code for an annotation whose class is unmapped/out-of-range (Req 1.7).
UNMAPPED_CLASS = "UNMAPPED_CLASS"


class SourceClassMapper:
    """Map non-unified source annotations to Unified_Class_Space ids (Req 1.3).

    The mapper is deliberately simple and **injectable**: by default it is the identity
    over the unified id set ``{0, 1, 2, 3}`` (a source id that is already a unified id
    maps to itself). Per-dataset overrides let a dataset that encodes classes differently
    -- for example UFPR-ALPR's line-2 vehicle codes, or any dataset-local class ids --
    remap its source ids onto unified ids before coverage is computed.

    ``overrides`` is keyed by dataset id; each value maps a *source* class id to a
    *unified* class id. An override that points at a non-unified id, or a source id that
    has neither an override nor an identity match in the unified set, is reported as
    unmapped (``map_class`` returns ``None``) so the caller can exclude the annotation and
    emit a diagnostic (Req 1.7).
    """

    def __init__(self, overrides: dict[str, dict[int, int]] | None = None) -> None:
        # Copy defensively so callers cannot mutate our mapping after construction.
        self._overrides: dict[str, dict[int, int]] = {
            dataset_id: dict(mapping) for dataset_id, mapping in (overrides or {}).items()
        }

    def map_class(self, dataset_id: str, source_class_id: int) -> int | None:
        """Return the unified class id for a source class id, or ``None`` if unmapped.

        Resolution order:

        1. A per-dataset override for ``source_class_id`` (only honoured when it targets a
           unified id; an override to a non-unified id is treated as unmapped).
        2. Identity, when ``source_class_id`` is itself a unified id.
        3. Otherwise ``None`` -- no mapping / out of range (Req 1.7).
        """
        dataset_overrides = self._overrides.get(dataset_id)
        if dataset_overrides is not None and source_class_id in dataset_overrides:
            mapped = dataset_overrides[source_class_id]
            return mapped if mapped in UNIFIED_IDS else None

        if source_class_id in UNIFIED_IDS:
            return source_class_id

        return None


@dataclass(frozen=True)
class SplitCoverage:
    """Present/absent unified-class partition for a single Dataset_Split (Req 1.4)."""

    dataset_id: str
    split: str
    present: frozenset[int]
    absent: frozenset[int]  # UNIFIED_IDS - present (Req 1.2)
    diagnostics: list[Diagnostic] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetCoverage:
    """Aggregated coverage for a dataset across all of its splits (Req 1.4, 1.5)."""

    dataset_id: str
    per_split: dict[str, SplitCoverage]
    unreadable: bool  # Req 1.5


@dataclass(frozen=True)
class _SplitScan:
    """Internal scan result carrying open-failure counts for unreadable detection."""

    coverage: SplitCoverage
    files_total: int
    files_unopenable: int


class CoverageAnalyzer:
    """Determine present/absent unified classes per dataset and split (Req 1)."""

    def __init__(self, mapper: SourceClassMapper | None = None) -> None:
        self._mapper = mapper if mapper is not None else SourceClassMapper()

    def analyze_split(
        self, dataset_id: str, split: str, label_files: list[str]
    ) -> SplitCoverage:
        """Analyze one split's label files and return its :class:`SplitCoverage`.

        ``present`` is the set of distinct mapped unified ids with at least one valid
        annotation (Req 1.1); ``absent`` is the explicit complement over
        :data:`~pseudo_labeling.unified.UNIFIED_IDS` (Req 1.2). Malformed files are
        excluded (Req 1.6) and unmapped/out-of-range annotations are dropped (Req 1.7),
        each with a diagnostic.
        """
        return self._scan_split(dataset_id, split, label_files).coverage

    def analyze_dataset(
        self, dataset_id: str, splits: dict[str, list[str]]
    ) -> DatasetCoverage:
        """Analyze every split of a dataset (Req 1.4) and flag unreadable datasets (Req 1.5).

        ``splits`` maps each split name to its list of label-file paths. When every
        provided label file across all splits is unopenable, the dataset is flagged
        ``unreadable`` (present sets are empty) and a dataset-identifying diagnostic is
        recorded (Req 1.5).
        """
        per_split: dict[str, SplitCoverage] = {}
        files_total = 0
        files_unopenable = 0

        for split, label_files in splits.items():
            scan = self._scan_split(dataset_id, split, label_files)
            per_split[split] = scan.coverage
            files_total += scan.files_total
            files_unopenable += scan.files_unopenable

        # A dataset is unreadable when it has label files but none of them could be opened
        # (Req 1.5). In that case every present set is already empty because nothing was
        # read; we add a dataset-level diagnostic identifying the dataset.
        unreadable = files_total > 0 and files_unopenable == files_total
        if unreadable:
            dataset_diagnostic = Diagnostic(
                code=UNREADABLE_DATASET,
                message=(
                    f"dataset {dataset_id!r} is unreadable: none of its "
                    f"{files_total} label file(s) could be opened"
                ),
                target=dataset_id,
            )
            for coverage in per_split.values():
                coverage.diagnostics.append(dataset_diagnostic)

        return DatasetCoverage(
            dataset_id=dataset_id,
            per_split=per_split,
            unreadable=unreadable,
        )

    def _scan_split(
        self, dataset_id: str, split: str, label_files: list[str]
    ) -> _SplitScan:
        """Scan a split's files, collecting present ids, diagnostics, and open-failures."""
        present: set[int] = set()
        diagnostics: list[Diagnostic] = []
        files_unopenable = 0

        for path in label_files:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    contents = handle.read()
            except OSError as exc:
                # File could not be opened -- feeds unreadable-dataset detection (Req 1.5).
                files_unopenable += 1
                diagnostics.append(
                    Diagnostic(
                        code=UNREADABLE_LABEL,
                        message=f"could not open label file {path!r}: {exc}",
                        target=path,
                    )
                )
                continue

            file_present = self._scan_file(dataset_id, path, contents, diagnostics)
            if file_present is not None:
                present.update(file_present)

        present_frozen = frozenset(present)
        coverage = SplitCoverage(
            dataset_id=dataset_id,
            split=split,
            present=present_frozen,
            absent=UNIFIED_IDS - present_frozen,  # Req 1.2 explicit complement
            diagnostics=diagnostics,
        )
        return _SplitScan(
            coverage=coverage,
            files_total=len(label_files),
            files_unopenable=files_unopenable,
        )

    def _scan_file(
        self,
        dataset_id: str,
        path: str,
        contents: str,
        diagnostics: list[Diagnostic],
    ) -> set[int] | None:
        """Parse one label file's contents.

        Returns the set of mapped unified ids found in the file, or ``None`` when the
        file is malformed and excluded in its entirety (Req 1.6). Blank/whitespace-only
        lines are ignored so trailing newlines do not mark a file malformed.
        """
        lines = [line for line in contents.splitlines() if line.strip()]

        parsed_classes: list[int] = []
        for line in lines:
            try:
                label = parse_line(line)
            except ValueError as exc:
                # Malformed class id / geometry -> exclude the whole file (Req 1.6).
                diagnostics.append(
                    Diagnostic(
                        code=MALFORMED_LABEL,
                        message=(
                            f"excluding malformed label file {path!r}: {exc}"
                        ),
                        target=path,
                    )
                )
                return None
            parsed_classes.append(label.class_id)

        file_present: set[int] = set()
        for source_class_id in parsed_classes:
            mapped = self._mapper.map_class(dataset_id, source_class_id)
            if mapped is None:
                # Unmapped source class or out-of-range id -> drop annotation (Req 1.7).
                diagnostics.append(
                    Diagnostic(
                        code=UNMAPPED_CLASS,
                        message=(
                            f"excluding annotation with unmapped/out-of-range class id "
                            f"{source_class_id!r} in {path!r}"
                        ),
                        target=path,
                    )
                )
                continue
            file_present.add(mapped)

        return file_present
