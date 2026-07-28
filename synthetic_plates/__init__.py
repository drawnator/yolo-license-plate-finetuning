"""
Synthetic Brazilian License Plate Generator
===========================================

Generates realistic synthetic Brazilian license plates (Mercosul + old models)
for data augmentation in YOLO-based license plate detection training.

Usage (CLI):
    python -m synthetic_plates generate --type mercosul --output plate.png
    python -m synthetic_plates generate --type all --count 20 --grid preview.jpg
    python -m synthetic_plates overlay --background image.jpg --output out.jpg
    python -m synthetic_plates build-dataset --backgrounds data/backgrounds/ --count 5000

Usage (API):
    from synthetic_plates import generate_plate, render_plate, overlay_on_background

For online training augmentation (requires albumentations):
    from synthetic_plates.augment import synthetic_plate_augmentation
"""

from synthetic_plates.plate_generator import generate_plate, render_plate, PlateRenderer
from synthetic_plates.plate_types import (
    PlateType,
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
    ALL_PLATE_TYPES,
    MERCOSUL_TYPES,
    OLD_TYPES,
    generate_plate_text,
)
from synthetic_plates.overlay import overlay_on_background, random_overlay_params, OverlayParams
from synthetic_plates.dataset_builder import build_dataset, SyntheticDatasetBuilder

__all__ = [
    "generate_plate",
    "render_plate",
    "PlateRenderer",
    "PlateType",
    "MERCOSUL_PARTICULAR",
    "MERCOSUL_ALUGUEL",
    "MERCOSUL_ESPECIAL",
    "MERCOSUL_OFICIAL",
    "MERCOSUL_DIPLOMATICA",
    "MERCOSUL_COLECAO",
    "CINZA",
    "VERMELHA",
    "VERDE",
    "AZUL",
    "PRETA",
    "DOURADA",
    "ALL_PLATE_TYPES",
    "MERCOSUL_TYPES",
    "OLD_TYPES",
    "generate_plate_text",
    "overlay_on_background",
    "random_overlay_params",
    "OverlayParams",
    "build_dataset",
    "SyntheticDatasetBuilder",
]
