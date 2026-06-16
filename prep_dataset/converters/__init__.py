"""Dataset format converters.

Each converter transforms a raw extracted dataset into YOLO format.
Converters are registered by name and invoked from the pipeline when
a dataset's config specifies a 'converter' field.
"""

from __future__ import annotations

from typing import Callable, Dict

from prep_dataset.converters.ufpr_alpr import convert_ufpr_alpr
from prep_dataset.converters.rodosol import convert_rodosol

# Registry of available converters: name -> callable(extract_path)
CONVERTERS: Dict[str, Callable[[str], None]] = {
    "ufpr_alpr": convert_ufpr_alpr,
    "rodosol": convert_rodosol,
}


def get_converter(name: str | None) -> Callable[[str], None] | None:
    """Look up a converter by name. Returns None if name is None or unknown."""
    if not name:
        return None
    return CONVERTERS.get(name)
