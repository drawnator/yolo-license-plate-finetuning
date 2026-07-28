"""Brazilian license plate type definitions.

Covers both Mercosul (current) and old (pre-2018) plate models.

Reference: https://pt.wikipedia.org/wiki/Placas_de_identificação_de_veículos_no_Brasil

Mercosul plates have a white background with a blue band on top.
Variants (all white background, varying text/border colors):
  - Particular:   black text, no border  (default)
  - Aluguel:      red text, red border   (commercial/rental)
  - Especial:     green text, green border
  - Oficial:      blue text, blue border
  - Diplomática:  gold text, gold border
  - Coleção:      black background, white text + white border

Old plates (pre-Mercosul, still in circulation):
  - Cinza:     gray background, black text  (particular)
  - Vermelha:  red background, white text   (commercial/rental)
  - Verde:     green background, white text (special)
  - Azul:      blue background, white text  (official)
  - Preta:     black background, white text (collection)
  - Dourada:   gold background, black text  (diplomatic)
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple


class PlateFormat(Enum):
    """License plate text format."""
    MERCOSUL = auto()      # ABC1D23
    OLD_ALPHA = auto()     # ABC1234
    OLD_ALPHA2 = auto()    # ABC-1234 (with hyphen)


@dataclass
class PlateType:
    """Definition of a Brazilian license plate type.

    Attributes:
        name: Human-readable name.
        format: Text format (Mercosul, old with or without hyphen).
        bg_color: Background color as (R, G, B) tuple.
        text_color: Text/holemark color as (R, G, B) tuple.
        border_color: Border color (None = same as bg or no border).
        holes: Whether the plate has mounting holes (4 circles).
        mercosul: Whether this is a Mercosul-format plate (blue band on top).
        weight: Probability weight for random selection.
    """
    name: str
    format: PlateFormat
    bg_color: Tuple[int, int, int]
    text_color: Tuple[int, int, int]
    border_color: Tuple[int, int, int] | None = None
    holes: bool = True
    mercosul: bool = False
    weight: float = 1.0

    @property
    def has_border(self) -> bool:
        """Whether this plate type draws an explicit border."""
        return self.border_color is not None


# ── Mercosul variants ────────────────────────────────────────────────
# All have white background, blue Mercosul band on top.
# The distinction is in text + border color.

MERCOSUL_PARTICULAR = PlateType(
    name="mercosul_particular",
    format=PlateFormat.MERCOSUL,
    bg_color=(255, 255, 255),
    text_color=(0, 0, 0),
    border_color=None,
    mercosul=True,
    weight=6.0,  # most common
)

MERCOSUL_ALUGUEL = PlateType(
    name="mercosul_aluguel",
    format=PlateFormat.MERCOSUL,
    bg_color=(255, 255, 255),
    text_color=(255, 0, 0),
    border_color=(255, 0, 0),
    mercosul=True,
    weight=1.5,
)

MERCOSUL_ESPECIAL = PlateType(
    name="mercosul_especial",
    format=PlateFormat.MERCOSUL,
    bg_color=(255, 255, 255),
    text_color=(0, 128, 0),
    border_color=(0, 128, 0),
    mercosul=True,
    weight=0.5,
)

MERCOSUL_OFICIAL = PlateType(
    name="mercosul_oficial",
    format=PlateFormat.MERCOSUL,
    bg_color=(255, 255, 255),
    text_color=(0, 0, 255),
    border_color=(0, 0, 255),
    mercosul=True,
    weight=0.5,
)

MERCOSUL_DIPLOMATICA = PlateType(
    name="mercosul_diplomatica",
    format=PlateFormat.MERCOSUL,
    bg_color=(255, 255, 255),
    text_color=(192, 160, 16),
    border_color=(192, 160, 16),
    mercosul=True,
    weight=0.1,
)

MERCOSUL_COLECAO = PlateType(
    name="mercosul_colecao",
    format=PlateFormat.MERCOSUL,
    bg_color=(0, 0, 0),
    text_color=(255, 255, 255),
    border_color=(255, 255, 255),
    mercosul=True,
    weight=0.1,
)

# ── Old (pre-Mercosul) models ────────────────────────────────────────

CINZA = PlateType(
    name="cinza",
    format=PlateFormat.OLD_ALPHA,
    bg_color=(192, 192, 192),
    text_color=(0, 0, 0),
    weight=3.0,
)

VERMELHA = PlateType(
    name="vermelha",
    format=PlateFormat.OLD_ALPHA,
    bg_color=(255, 0, 0),
    text_color=(255, 255, 255),
    weight=0.8,
)

VERDE = PlateType(
    name="verde",
    format=PlateFormat.OLD_ALPHA,
    bg_color=(0, 128, 0),
    text_color=(255, 255, 255),
    weight=0.3,
)

AZUL = PlateType(
    name="azul",
    format=PlateFormat.OLD_ALPHA,
    bg_color=(0, 0, 255),
    text_color=(255, 255, 255),
    weight=0.3,
)

PRETA = PlateType(
    name="preta",
    format=PlateFormat.OLD_ALPHA,
    bg_color=(0, 0, 0),
    text_color=(255, 255, 255),
    weight=0.1,
)

DOURADA = PlateType(
    name="dourada",
    format=PlateFormat.OLD_ALPHA,
    bg_color=(192, 160, 16),
    text_color=(0, 0, 0),
    weight=0.1,
)

# ── Collections ──────────────────────────────────────────────────────

ALL_PLATE_TYPES: list[PlateType] = [
    MERCOSUL_PARTICULAR,
    MERCOSUL_ALUGUEL,
    MERCOSUL_ESPECIAL,
    MERCOSUL_OFICIAL,
    MERCOSUL_DIPLOMATICA,
    MERCOSUL_COLECAO,
    CINZA,
    VERMELHA,
    VERDE,
    AZUL,
    PRETA,
    DOURADA,
]

MERCOSUL_TYPES: list[PlateType] = [
    t for t in ALL_PLATE_TYPES if t.mercosul
]

OLD_TYPES: list[PlateType] = [
    t for t in ALL_PLATE_TYPES if not t.mercosul
]


def generate_plate_text(fmt: PlateFormat) -> str:
    """Generate a random Brazilian plate text in the given format.

    Mercosul format ``ABC1D23``:
        Positions 1-3: uppercase letters (A-Z)
        Position 4: digit (0-9)
        Position 5: uppercase letter
        Positions 6-7: digits

    Old format ``ABC1234`` / ``ABC-1234``:
        Positions 1-3: uppercase letters
        Positions 4-7: digits
    """
    if fmt == PlateFormat.MERCOSUL:
        letters1 = "".join(random.choices(string.ascii_uppercase, k=3))
        digit1 = random.choice(string.digits)
        letter2 = random.choice(string.ascii_uppercase)
        digits2 = "".join(random.choices(string.digits, k=2))
        return f"{letters1}{digit1}{letter2}{digits2}"
    elif fmt in (PlateFormat.OLD_ALPHA, PlateFormat.OLD_ALPHA2):
        letters = "".join(random.choices(string.ascii_uppercase, k=3))
        digits = "".join(random.choices(string.digits, k=4))
        if fmt == PlateFormat.OLD_ALPHA2:
            return f"{letters}-{digits}"
        return f"{letters}{digits}"
    raise ValueError(f"Unknown plate format: {fmt}")


# Simple validation regexes
_MERCOSUL_RE = re.compile(r"^[A-Z]{3}\d[A-Z]\d{2}$")
_OLD_RE = re.compile(r"^[A-Z]{3}-?\d{4}$")


def is_valid_plate_text(text: str, fmt: PlateFormat | None = None) -> bool:
    """Check if text is valid for the given (or auto-detected) format."""
    if fmt == PlateFormat.MERCOSUL or (fmt is None and _MERCOSUL_RE.match(text)):
        return bool(_MERCOSUL_RE.match(text))
    if fmt in (PlateFormat.OLD_ALPHA, PlateFormat.OLD_ALPHA2) or (fmt is None):
        return bool(_OLD_RE.match(text))
    return False


def random_plate_type(
    mercosul_only: bool = False,
    old_only: bool = False,
    weights: dict[str, float] | None = None,
) -> PlateType:
    """Pick a random plate type, weighted by prevalence.

    Args:
        mercosul_only: Only return Mercosul types.
        old_only: Only return old (pre-Mercosul) types.
        weights: Optional per-type weight overrides, keyed by ``PlateType.name``.
    """
    if mercosul_only and old_only:
        raise ValueError("Cannot set both mercosul_only and old_only")

    if mercosul_only:
        pool = MERCOSUL_TYPES
    elif old_only:
        pool = OLD_TYPES
    else:
        pool = ALL_PLATE_TYPES

    if weights:
        w = [weights.get(t.name, t.weight) for t in pool]
    else:
        w = [t.weight for t in pool]

    return random.choices(pool, weights=w, k=1)[0]


# ── Rendering constants (fractions of plate width) ───────────────────
# Plate aspect ratio: 400 × 130 mm (≈ 3.08:1). We render at 400×130 px
# and scale down during overlay.

PLATE_WIDTH = 400
PLATE_HEIGHT = 130
MERCOSUL_BAND_HEIGHT = 20  # px, blue band on top

# Margins relative to plate edges
MARGIN_LEFT = 0.04    # fraction of plate width
MARGIN_RIGHT = 0.04
MARGIN_TOP = 0.04 + (MERCOSUL_BAND_HEIGHT / PLATE_HEIGHT if False else 0.04)
MARGIN_BOTTOM = 0.04

# Character spacing
LETTER_SPACING = 0.02   # fraction of plate width between chars

# Hole positions (fraction of plate width/height, from top-left)
HOLE_OFFSET_X = 0.06
HOLE_OFFSET_Y = 0.65   # vertical center-ish
HOLE_RADIUS = 0.025     # fraction of plate height → ~3.25px at 130px

# Border thickness
BORDER_THICKNESS = 3  # px

# Mercosul blue band color
MERCOSUL_BAND_COLOR = (0, 51, 153)  # Blue (#003399)

# Mercosul band text ("BRASIL" + flag placeholder)
MERCOSUL_BAND_TEXT_COLOR = (255, 255, 255)
MERCOSUL_BAND_FLAG_COLOR = (0, 155, 0)   # Green
MERCOSUL_BAND_FLAG_COLOR2 = (255, 255, 0)  # Yellow (for losango)