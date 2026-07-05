"""Unified_Class_Space constants and ``data.yaml`` loading (Req 9.4).

The unified four-class label space (``plate=0``, ``face=1``, ``car=2``,
``motorcycle=3``) is defined by the project ``data.yaml`` via its ``nc`` and ``names``
keys. This module is the single place that reads that file and exposes the id->name
mapping plus a membership check over the unified id set.
"""

from __future__ import annotations

import yaml

# The unified class ids over which coverage present/absent partitions are computed.
UNIFIED_IDS: frozenset[int] = frozenset({0, 1, 2, 3})


def load_unified_space(data_yaml_path: str) -> dict[int, str]:
    """Load the ``{id: name}`` Unified_Class_Space mapping from a ``data.yaml`` (Req 9.4).

    Reads the ``nc`` (class count) and ``names`` (ordered class names) keys and returns
    a mapping from each class id to its name, where the id is the position of the name
    in the ``names`` list.

    Raises:
        ValueError: if ``nc`` does not equal ``len(names)``.
    """
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    nc = data["nc"]
    names = data["names"]

    if nc != len(names):
        raise ValueError(
            f"data.yaml nc ({nc}) does not match number of names ({len(names)}) "
            f"in {data_yaml_path!r}"
        )

    return {i: name for i, name in enumerate(names)}


def is_unified_id(class_id: int) -> bool:
    """Return ``True`` when ``class_id`` belongs to the Unified_Class_Space."""
    return class_id in UNIFIED_IDS
