"""Brazilian license plate image renderer using PIL/Pillow.

Renders high-resolution plate images (400×130 px) with:
- Correct background color
- Blue Mercosul band with "BRASIL" and flag lozenge
- Character text with proper spacing
- Mounting holes (4 circles)
- Optional border for some plate types
"""

from __future__ import annotations

import logging
import os
import random
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

from synthetic_plates.plate_types import (
    BORDER_THICKNESS,
    HOLE_OFFSET_X,
    HOLE_OFFSET_Y,
    HOLE_RADIUS,
    LETTER_SPACING,
    MARGIN_BOTTOM,
    MARGIN_LEFT,
    MARGIN_RIGHT,
    MARGIN_TOP,
    MERCOSUL_BAND_COLOR,
    MERCOSUL_BAND_HEIGHT,
    MERCOSUL_BAND_TEXT_COLOR,
    MERCOSUL_BAND_FLAG_COLOR,
    MERCOSUL_BAND_FLAG_COLOR2,
    PLATE_HEIGHT,
    PLATE_WIDTH,
    PlateFormat,
    PlateType,
    generate_plate_text,
    random_plate_type,
)

logger = logging.getLogger(__name__)

# ── Font discovery ───────────────────────────────────────────────────

_FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")

# Priority: FE-Schrift (correct for Mercosul) → DejaVu Sans Mono → system fallback
_FONT_CANDIDATES = [
    os.path.join(_FONTS_DIR, "FE-Schrift.ttf"),
    os.path.join(_FONTS_DIR, "FESchrift.ttf"),
    os.path.join(_FONTS_DIR, "DejaVuSansMono-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
]

# Add matplotlib's bundled DejaVu fonts (present in Nix/conda envs)
try:
    import matplotlib

    _mpl_font_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    _FONT_CANDIDATES.extend(
        [
            os.path.join(_mpl_font_dir, "DejaVuSansMono-Bold.ttf"),
            os.path.join(_mpl_font_dir, "DejaVuSansMono-BoldOblique.ttf"),
            os.path.join(_mpl_font_dir, "DejaVuSans-Bold.ttf"),
            os.path.join(_mpl_font_dir, "DejaVuSansMono.ttf"),
            os.path.join(_mpl_font_dir, "DejaVuSans.ttf"),
        ]
    )
except ImportError:
    pass


def _find_font() -> str:
    """Return the first available font path from the candidate list."""
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    # last resort: None → PIL default font
    return ""


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font at the given size, falling back to PIL default."""
    font_path = _find_font()
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── Main rendering ───────────────────────────────────────────────────

class PlateRenderer:
    """Renders Brazilian license plates onto PIL Images."""

    def __init__(self, width: int = PLATE_WIDTH, height: int = PLATE_HEIGHT):
        self.width = width
        self.height = height

    def render(self, plate_type: PlateType, plate_text: str | None = None) -> Image.Image:
        """Render a single license plate image.

        Args:
            plate_type: The plate type definition (colors, format, etc.).
            plate_text: Optional plate text. If None, a random valid text is generated.

        Returns:
            PIL Image in RGB mode.
        """
        if plate_text is None:
            plate_text = generate_plate_text(plate_type.format)

        img = Image.new("RGB", (self.width, self.height), plate_type.bg_color)
        draw = ImageDraw.Draw(img)

        # 1. Draw Mercosul blue band (top)
        if plate_type.mercosul:
            self._draw_mercosul_band(draw)

        # 2. Draw border (if applicable)
        if plate_type.has_border and plate_type.border_color:
            self._draw_border(draw, plate_type.border_color)

        # 3. Draw plate text
        text_area_top = MERCOSUL_BAND_HEIGHT if plate_type.mercosul else 0
        self._draw_text(draw, plate_text, plate_type.text_color, text_area_top)

        # 4. Draw mounting holes
        if plate_type.holes:
            self._draw_holes(draw)

        return img

    def _draw_mercosul_band(self, draw: ImageDraw.Draw) -> None:
        """Draw the blue Mercosul band at the top with 'BRASIL' and a flag lozenge."""
        # Blue band
        draw.rectangle([0, 0, self.width, MERCOSUL_BAND_HEIGHT], fill=MERCOSUL_BAND_COLOR)

        # "BRASIL" text — small, centered horizontally in the band
        font_size = max(9, MERCOSUL_BAND_HEIGHT - 6)
        font = _load_font(font_size)
        text = "BRASIL"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (self.width - text_w) // 2
        text_y = (MERCOSUL_BAND_HEIGHT - text_h) // 2
        draw.text((text_x, text_y), text, fill=MERCOSUL_BAND_TEXT_COLOR, font=font)

        # Small Brazilian flag lozenge (yellow diamond on green, blue circle w/ stars)
        # Simplified: just a small green rectangle with yellow lozenge
        flag_size = MERCOSUL_BAND_HEIGHT - 4
        flag_x = self.width - flag_size - 6
        flag_y = 2

        # Green background
        draw.rectangle([flag_x, flag_y, flag_x + flag_size, flag_y + flag_size],
                       fill=MERCOSUL_BAND_FLAG_COLOR)

        # Yellow lozenge (rotated square)
        cx = flag_x + flag_size // 2
        cy = flag_y + flag_size // 2
        half = flag_size // 3
        draw.polygon([
            (cx, cy - half),
            (cx + half, cy),
            (cx, cy + half),
            (cx - half, cy),
        ], fill=MERCOSUL_BAND_FLAG_COLOR2)

    def _draw_border(self, draw: ImageDraw.Draw, color: Tuple[int, int, int]) -> None:
        """Draw a border rectangle inset from the plate edges."""
        draw.rectangle(
            [BORDER_THICKNESS // 2, BORDER_THICKNESS // 2,
             self.width - BORDER_THICKNESS // 2, self.height - BORDER_THICKNESS // 2],
            outline=color,
            width=BORDER_THICKNESS,
        )

    def _draw_text(
        self,
        draw: ImageDraw.Draw,
        text: str,
        color: Tuple[int, int, int],
        top_offset: int = 0,
    ) -> None:
        """Draw the license plate characters centered in the available area."""
        available_height = self.height - top_offset
        # Use ~55% of available height for character height
        font_size = int(available_height * 0.55)
        font = _load_font(font_size)

        # Calculate total text width
        chars = list(text)
        char_widths = []
        total_text_w = 0
        for ch in chars:
            bbox = draw.textbbox((0, 0), ch, font=font)
            char_widths.append(bbox[2] - bbox[0])
            total_text_w += bbox[2] - bbox[0]

        # Add spacing between chars
        spacing_px = int(self.width * LETTER_SPACING)
        total_w = total_text_w + spacing_px * (len(chars) - 1)

        # Center in the text area (below the Mercosul band)
        start_x = (self.width - total_w) // 2
        text_area_center_y = top_offset + available_height // 2

        x = start_x
        for i, ch in enumerate(chars):
            bbox = draw.textbbox((0, 0), ch, font=font)
            ch_h = bbox[3] - bbox[1]
            ch_y = text_area_center_y - ch_h // 2

            # Subtle vertical alignment — some fonts have odd baseline
            draw.text((x, ch_y), ch, fill=color, font=font)
            x += char_widths[i] + spacing_px

    def _draw_holes(self, draw: ImageDraw.Draw) -> None:
        """Draw 4 mounting holes (small circles) near the corners."""
        hole_r = int(self.width * HOLE_RADIUS)
        hole_x_offset = int(self.width * HOLE_OFFSET_X)
        hole_y = int(self.height * HOLE_OFFSET_Y)

        positions = [
            (hole_x_offset, hole_y),
            (self.width - hole_x_offset, hole_y),
        ]

        for hx, hy in positions:
            draw.ellipse(
                [hx - hole_r, hy - hole_r, hx + hole_r, hy + hole_r],
                fill=(60, 60, 60),
            )


# ── Convenience functions ────────────────────────────────────────────

# Module-level renderer instance
_renderer = PlateRenderer()


def render_plate(
    plate_type: PlateType | None = None,
    plate_text: str | None = None,
    mercosul_only: bool = False,
    old_only: bool = False,
) -> Image.Image:
    """Render a single license plate.

    Args:
        plate_type: Specific plate type. If None, a random type is chosen.
        plate_text: Plate text. If None, auto-generated based on format.
        mercosul_only: If plate_type is None, only pick from Mercosul types.
        old_only: If plate_type is None, only pick from old types.

    Returns:
        Rendered PIL Image.
    """
    if plate_type is None:
        plate_type = random_plate_type(mercosul_only=mercosul_only, old_only=old_only)
    return _renderer.render(plate_type, plate_text)


def generate_plate(
    output_path: str,
    plate_type: PlateType | None = None,
    plate_text: str | None = None,
) -> None:
    """Generate and save a single plate image to disk."""
    img = render_plate(plate_type=plate_type, plate_text=plate_text)
    img.save(output_path)
    logger.info("Saved plate to %s", output_path)


def generate_grid(
    count: int = 20,
    cols: int = 5,
    mercosul_only: bool = False,
    old_only: bool = False,
) -> Image.Image:
    """Generate a grid of random plates for preview/inspection.

    Args:
        count: Total number of plates to render.
        cols: Number of columns in the grid.
        mercosul_only: Only generate Mercosul plates.
        old_only: Only generate old-format plates.

    Returns:
        PIL Image containing the grid.
    """
    rows = (count + cols - 1) // cols
    plate_w = PLATE_WIDTH
    plate_h = PLATE_HEIGHT
    gap = 10

    grid = Image.new(
        "RGB",
        (cols * plate_w + (cols + 1) * gap, rows * plate_h + (rows + 1) * gap),
        (240, 240, 240),
    )

    for i in range(count):
        plate = render_plate(mercosul_only=mercosul_only, old_only=old_only)
        row = i // cols
        col = i % cols
        x = gap + col * (plate_w + gap)
        y = gap + row * (plate_h + gap)
        grid.paste(plate, (x, y))

    return grid


def available_fonts() -> list[str]:
    """List all discovered font paths (for debugging)."""
    found: list[str] = []
    for path in _FONT_CANDIDATES:
        exists = os.path.isfile(path)
        found.append(f"{'✓' if exists else '✗'} {path}")
    return found