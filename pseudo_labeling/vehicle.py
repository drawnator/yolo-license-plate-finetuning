"""Vehicle class resolution for RodoSol-ALPR images (Req 6).

The RodoSol-ALPR dataset encodes the vehicle type in the image folder name: images live
under directories named ``cars-*`` (for example ``cars-br`` / ``cars-me``) or
``motorcycles-*`` (for example ``motorcycles-br`` / ``motorcycles-me``). This module maps
the **immediate parent directory name** of an image to the unified vehicle class id so that
vehicle pseudo-labels for RodoSol images are assigned the correct class.

This module is **pure**: it inspects the given path string only and never touches the
filesystem. Paths are treated as POSIX-style (forward-slash separated), matching the dataset
layout under ``datasets/``.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from .models import UnifiedClass

#: Case-sensitive prefix identifying a car folder (Req 6.1).
CAR_PREFIX = "cars-"
#: Case-sensitive prefix identifying a motorcycle folder (Req 6.2).
MOTORCYCLE_PREFIX = "motorcycles-"


class VehicleResolutionError(Exception):
    """Raised when a RodoSol vehicle class id cannot be resolved from the image path.

    This occurs when the immediate parent directory name matches neither the ``cars-`` nor
    the ``motorcycles-`` prefix, matches both, or resolution otherwise fails (Req 6.4).
    """


def resolve_vehicle_class(image_path: str) -> int:
    """Resolve the unified vehicle class id from a RodoSol-ALPR image path (Req 6.1, 6.2).

    Inspects the **immediate parent directory name** of ``image_path``:

    - A case-sensitive ``cars-`` prefix resolves to :attr:`UnifiedClass.CAR` (``2``).
    - A case-sensitive ``motorcycles-`` prefix resolves to
      :attr:`UnifiedClass.MOTORCYCLE` (``3``).

    Args:
        image_path: POSIX-style path to a RodoSol-ALPR image.

    Returns:
        The resolved unified vehicle class id (``2`` for car, ``3`` for motorcycle).

    Raises:
        VehicleResolutionError: if the immediate parent directory name matches neither
            prefix, matches both prefixes, or cannot otherwise be resolved (Req 6.4).
    """
    parts = PurePosixPath(image_path).parts
    # Need at least a parent directory and a filename to have an "immediate parent".
    if len(parts) < 2:
        raise VehicleResolutionError(
            f"cannot resolve vehicle class: no parent directory in image path: {image_path!r}"
        )

    parent_name = parts[-2]
    is_car = parent_name.startswith(CAR_PREFIX)
    is_motorcycle = parent_name.startswith(MOTORCYCLE_PREFIX)

    # Guard defensively against a name matching both prefixes (not possible for a single
    # name today, but treated as unresolvable per Req 6.4).
    if is_car and is_motorcycle:
        raise VehicleResolutionError(
            f"cannot resolve vehicle class: parent directory {parent_name!r} matches both "
            f"{CAR_PREFIX!r} and {MOTORCYCLE_PREFIX!r} prefixes"
        )
    if is_car:
        return int(UnifiedClass.CAR)
    if is_motorcycle:
        return int(UnifiedClass.MOTORCYCLE)

    raise VehicleResolutionError(
        f"cannot resolve vehicle class: parent directory {parent_name!r} matches neither "
        f"{CAR_PREFIX!r} nor {MOTORCYCLE_PREFIX!r} prefix"
    )
