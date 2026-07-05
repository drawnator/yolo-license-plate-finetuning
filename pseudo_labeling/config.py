"""Configuration model, merge-mode/output-target resolution, and validation.

This module implements the *core* of ``config.py`` from the design: the configuration
dataclasses plus the pure resolution/validation helpers. Config *loading* (from a YAML
file and CLI flags) and the CLI itself live in ``__main__.py`` (task 17) and are not
implemented here.

All validation happens *before* any merge, so an invalid configuration leaves the
datasets untouched (Req 4.6, 10.6, 12.8, 12.9).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum

from .models import Diagnostic


class MergeMode(str, Enum):
    """Merge_Mode selection (Req 12.1). Default is REVIEW (Req 12.2).

    - ``REVIEW`` (Review_Mode): produce the Audit_Report and merge only after approval.
    - ``AUTO_MERGE`` (Auto_Merge_Mode): merge directly, no approval gate.
    """

    REVIEW = "review"
    AUTO_MERGE = "auto"


class OutputTarget(str, Enum):
    """Output_Target selection (Req 13.1). Default is IN_PLACE (Req 13.2).

    Orthogonal to :class:`MergeMode`.

    - ``IN_PLACE`` (In_Place_Target): write into the dataset's own labels dir.
    - ``LABEL_SET`` (Label_Set_Target): write into a separate label-set tree.
    """

    IN_PLACE = "in-place"
    LABEL_SET = "label-set"


@dataclass
class ThresholdConfig:
    """Confidence_Threshold configuration (Req 4.1, 4.3)."""

    default: float | None = None  # Req 4.3
    per_class: dict[int, float] = field(default_factory=dict)  # Req 4.1


@dataclass
class PseudoLabelingConfig:
    """Full run configuration for the Pseudo_Labeling_System."""

    data_yaml: str = "./data.yaml"
    datasets_root: str = "./datasets"
    base_model: str | None = None  # Req 2.3/2.7
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    dup_iou_threshold: float = 0.5  # Req 5.4
    # Merge_Mode selection replaces the old review_mode boolean.
    merge_mode: MergeMode = MergeMode.REVIEW  # Review_Mode default (Req 12.1, 12.2)
    # Raw, pre-validation value as supplied by config/CLI, used to detect ambiguous or
    # unrecognized merge-mode configuration (Req 12.8, 12.9). May be a str, list, or None.
    merge_mode_raw: object | None = None
    # Non_Interactive_Execution: auto-detected from absence of a TTY; overridable.
    non_interactive: bool | None = None  # None -> auto-detect via stdin.isatty() (Req 12.5)
    # Pre_Authorized_Approval: authorizes merging in Review_Mode without interactive confirm.
    assume_yes: bool = False  # --yes / --assume-yes (Req 12.6)
    # Output_Target selection (orthogonal to Merge_Mode). Default In_Place_Target (Req 13.1, 13.2).
    output_target: OutputTarget = OutputTarget.IN_PLACE
    label_set_id: str | None = None  # explicit id to (re)generate to same location (Req 14.5)
    label_set_name: str | None = None  # operator-supplied name -> derives label_set_id (Req 14.1)
    label_set_store_root: str = "datasets/label_sets"  # Label_Set_Store root (Req 14.1)
    package_label_set: bool = False  # --package/--archive: emit Label_Set_Archive (Req 14.4)
    label_set_archive_path: str | None = None  # optional archive destination
    seed: int | None = None  # Req 9.2/9.3
    run_id: str | None = None


class AmbiguousMergeModeError(Exception):
    """Raised when both Review_Mode and Auto_Merge_Mode are configured (Req 12.8)."""


class UnrecognizedMergeModeError(Exception):
    """Raised when the Merge_Mode value is neither 'review' nor 'auto' (Req 12.9)."""


# Accepted raw merge-mode tokens, mapped to their resolved MergeMode.
_MERGE_MODE_TOKENS: dict[str, MergeMode] = {
    MergeMode.REVIEW.value: MergeMode.REVIEW,
    MergeMode.AUTO_MERGE.value: MergeMode.AUTO_MERGE,
}


def _normalize_merge_tokens(raw: object) -> list[str]:
    """Normalize a raw merge-mode config value into a list of lowercase string tokens.

    Accepts a single string, a MergeMode, or a list/tuple/set of those. Used to detect
    ambiguous (multiple distinct modes) or unrecognized configurations.
    """
    if raw is None:
        return []
    if isinstance(raw, MergeMode):
        return [raw.value]
    if isinstance(raw, str):
        token = raw.strip().lower()
        return [token] if token else []
    if isinstance(raw, (list, tuple, set)):
        tokens: list[str] = []
        for item in raw:
            tokens.extend(_normalize_merge_tokens(item))
        return tokens
    # Any other type is treated as a single unrecognized token.
    return [str(raw).strip().lower()]


def resolve_merge_mode(cfg: PseudoLabelingConfig) -> MergeMode:
    """Resolve the effective Merge_Mode from ``cfg.merge_mode_raw`` / ``cfg.merge_mode``.

    - Unconfigured -> :attr:`MergeMode.REVIEW` (Req 12.2).
    - Ambiguous (both Review and Auto configured for the same run) ->
      :class:`AmbiguousMergeModeError` (Req 12.8).
    - Value that is neither 'review' nor 'auto' -> :class:`UnrecognizedMergeModeError`
      (Req 12.9).
    """
    # When a raw value is present it is authoritative for ambiguity/unrecognized detection.
    if cfg.merge_mode_raw is not None:
        tokens = _normalize_merge_tokens(cfg.merge_mode_raw)
        if not tokens:
            return MergeMode.REVIEW  # Req 12.2

        distinct = {t for t in tokens}
        unrecognized = distinct - set(_MERGE_MODE_TOKENS)
        if unrecognized:
            raise UnrecognizedMergeModeError(
                "Unrecognized Merge_Mode value(s): "
                f"{sorted(unrecognized)}; expected 'review' or 'auto'."
            )
        if len(distinct) > 1:
            raise AmbiguousMergeModeError(
                "Ambiguous Merge_Mode configuration: both Review_Mode and "
                "Auto_Merge_Mode were configured for the same run."
            )
        return _MERGE_MODE_TOKENS[next(iter(distinct))]

    # No raw value: fall back to the typed field, defaulting to Review_Mode.
    if cfg.merge_mode is None:
        return MergeMode.REVIEW  # Req 12.2
    if isinstance(cfg.merge_mode, MergeMode):
        return cfg.merge_mode
    # Defensive: a plain string slipped into the typed field.
    tokens = _normalize_merge_tokens(cfg.merge_mode)
    if not tokens:
        return MergeMode.REVIEW
    token = tokens[0]
    if token not in _MERGE_MODE_TOKENS:
        raise UnrecognizedMergeModeError(
            f"Unrecognized Merge_Mode value: {token!r}; expected 'review' or 'auto'."
        )
    return _MERGE_MODE_TOKENS[token]


def resolve_non_interactive(cfg: PseudoLabelingConfig) -> bool:
    """Return the effective Non_Interactive_Execution flag (Req 12.5).

    Honors ``cfg.non_interactive`` when explicitly set; otherwise auto-detects: True when
    no interactive terminal is attached (e.g. inside Docker), i.e. ``not sys.stdin.isatty()``.
    """
    if cfg.non_interactive is not None:
        return cfg.non_interactive

    stdin = getattr(sys, "stdin", None)
    isatty = getattr(stdin, "isatty", None)
    if not callable(isatty):
        # No usable stdin -> treat as non-interactive.
        return True
    try:
        return not isatty()
    except (ValueError, OSError):
        # Closed/detached stream -> treat as non-interactive.
        return True


def _threshold_out_of_range(value: float) -> bool:
    """True when a confidence threshold is outside the inclusive range [0.0, 1.0]."""
    return not (0.0 <= value <= 1.0)


def validate(cfg: PseudoLabelingConfig) -> list[Diagnostic]:
    """Validate the configuration before any merge, leaving datasets unchanged.

    Collects (does not raise) diagnostics for:
    - Confidence_Threshold values outside the inclusive range [0.0, 1.0], including the
      default and every per-class threshold (Req 4.6).
    - Ambiguous merge-mode configuration (Req 12.8) and unrecognized merge-mode values
      (Req 12.9); these are surfaced as diagnostics here even though
      :func:`resolve_merge_mode` raises for control flow.
    """
    diagnostics: list[Diagnostic] = []

    # --- Confidence thresholds (Req 4.6) ---
    thresholds = cfg.thresholds
    if thresholds is not None:
        if thresholds.default is not None and _threshold_out_of_range(thresholds.default):
            diagnostics.append(
                Diagnostic(
                    code="THRESHOLD_OUT_OF_RANGE",
                    message=(
                        f"Default Confidence_Threshold {thresholds.default} is out of the "
                        "allowed range 0.0 to 1.0 inclusive."
                    ),
                    target="thresholds.default",
                )
            )
        for class_id, value in thresholds.per_class.items():
            if _threshold_out_of_range(value):
                diagnostics.append(
                    Diagnostic(
                        code="THRESHOLD_OUT_OF_RANGE",
                        message=(
                            f"Confidence_Threshold {value} for class {class_id} is out of "
                            "the allowed range 0.0 to 1.0 inclusive."
                        ),
                        target=f"thresholds.per_class[{class_id}]",
                    )
                )

    # --- Merge-mode configuration (Req 12.8, 12.9) ---
    try:
        resolve_merge_mode(cfg)
    except AmbiguousMergeModeError as exc:
        diagnostics.append(
            Diagnostic(code="AMBIGUOUS_MERGE_MODE", message=str(exc), target="merge_mode")
        )
    except UnrecognizedMergeModeError as exc:
        diagnostics.append(
            Diagnostic(code="UNRECOGNIZED_MERGE_MODE", message=str(exc), target="merge_mode")
        )

    return diagnostics
