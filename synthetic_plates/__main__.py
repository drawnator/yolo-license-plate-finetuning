"""CLI for the synthetic plate generator.

Subcommands:
    generate       Render standalone plate images.
    overlay        Overlay plates onto a single background image.
    build-dataset  Generate a complete YOLO dataset from backgrounds.
    grid           Generate a preview grid of plates.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from synthetic_plates.plate_generator import (
    generate_grid,
    generate_plate,
    render_plate,
    available_fonts,
)
from synthetic_plates.plate_types import (
    ALL_PLATE_TYPES,
    MERCOSUL_TYPES,
    OLD_TYPES,
    PLATE_HEIGHT,
    PLATE_WIDTH,
    PlateFormat,
    PlateType,
    generate_plate_text,
    random_plate_type,
)
from synthetic_plates.overlay import (
    overlay_on_background,
    random_overlay_params,
    OverlayParams,
)
from synthetic_plates.dataset_builder import build_dataset

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Subcommand: generate ─────────────────────────────────────────────


def _cmd_generate(args: argparse.Namespace) -> int:
    """Generate one or more plate images."""
    _setup_logging(args.verbose)

    if args.type == "all":
        types = ALL_PLATE_TYPES
    elif args.type == "mercosul":
        types = MERCOSUL_TYPES
    elif args.type == "old":
        types = OLD_TYPES
    else:
        # Single type by name
        match = [t for t in ALL_PLATE_TYPES if t.name == args.type]
        if not match:
            print(f"Unknown plate type: {args.type}", file=sys.stderr)
            print(f"Available: {[t.name for t in ALL_PLATE_TYPES]}", file=sys.stderr)
            return 1
        types = match

    if args.grid:
        # Generate a grid preview
        img = generate_grid(
            count=args.count,
            cols=args.cols,
            mercosul_only=(args.type == "mercosul"),
            old_only=(args.type == "old"),
        )
        img.save(args.output)
        print(f"Grid saved to {args.output} ({args.count} plates, {args.cols} cols)")
    else:
        # Generate individual plates
        os.makedirs(args.output, exist_ok=True)
        for i in range(args.count):
            pt = random_plate_type()
            text = generate_plate_text(pt.format)
            path = os.path.join(args.output, f"plate_{i:04d}_{pt.name}_{text}.png")
            generate_plate(path, plate_type=pt, plate_text=text)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{args.count}...")
        print(f"Generated {args.count} plates in {args.output}")

    return 0


# ── Subcommand: overlay ──────────────────────────────────────────────


def _cmd_overlay(args: argparse.Namespace) -> int:
    """Overlay a synthetic plate onto a background image."""
    import cv2

    _setup_logging(args.verbose)

    background = cv2.imread(args.background)
    if background is None:
        print(f"Could not read background: {args.background}", file=sys.stderr)
        return 1

    if args.type:
        match = [t for t in ALL_PLATE_TYPES if t.name == args.type]
        plate_type = match[0] if match else None
    else:
        plate_type = None

    params = OverlayParams()
    composited, label = overlay_on_background(
        background=background,
        plate_type=plate_type,
        params=params,
    )

    cv2.imwrite(args.output, composited)
    cls_id, xc, yc, bw, bh = label
    print(f"Saved: {args.output}")
    print(f"YOLO label: {int(cls_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return 0


# ── Subcommand: build-dataset ────────────────────────────────────────


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    """Build a complete YOLO dataset."""
    _setup_logging(args.verbose)

    if not os.path.isdir(args.backgrounds):
        print(f"Background directory not found: {args.backgrounds}", file=sys.stderr)
        return 1

    yaml_path = build_dataset(
        background_dir=args.backgrounds,
        output_dir=args.output,
        image_count=args.count,
        plates_per_image=args.plates_per_image,
        max_plates_per_image=args.max_plates,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    print(f"Dataset built: {args.output}")
    print(f"data.yaml: {yaml_path}")
    return 0


# ── Subcommand: grid ─────────────────────────────────────────────────


def _cmd_grid(args: argparse.Namespace) -> int:
    """Generate a grid preview of plate types."""
    _setup_logging(args.verbose)

    mercosul_only = args.type == "mercosul"
    old_only = args.type == "old"

    img = generate_grid(
        count=args.count,
        cols=args.cols,
        mercosul_only=mercosul_only,
        old_only=old_only,
    )
    img.save(args.output)
    print(f"Preview grid saved to {args.output}")
    return 0


# ── Subcommand: fonts ────────────────────────────────────────────────


def _cmd_fonts(args: argparse.Namespace) -> int:
    """Show detected fonts."""
    fonts = available_fonts()
    for f in fonts:
        print(f)
    return 0


# ── Argument parsing ─────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthetic_plates",
        description="Synthetic Brazilian License Plate Generator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    p_gen = sub.add_parser("generate", help="Generate plate images")
    p_gen.add_argument("--type", default="all",
                       choices=["all", "mercosul", "old"] +
                               [t.name for t in ALL_PLATE_TYPES],
                       help="Plate type(s) to generate")
    p_gen.add_argument("--count", type=int, default=10,
                       help="Number of plates to generate")
    p_gen.add_argument("--output", default="synthetic_plates_output",
                       help="Output directory or file (if --grid)")
    p_gen.add_argument("--grid", action="store_true",
                       help="Generate a single grid image instead of individual files")
    p_gen.add_argument("--cols", type=int, default=5,
                       help="Columns for grid layout")
    p_gen.add_argument("--verbose", "-v", action="store_true")
    p_gen.set_defaults(func=_cmd_generate)

    # --- overlay ---
    p_ov = sub.add_parser("overlay", help="Overlay plate onto one background")
    p_ov.add_argument("--background", required=True,
                      help="Path to background image")
    p_ov.add_argument("--type", default=None,
                      help="Plate type name (random if omitted)")
    p_ov.add_argument("--output", default="overlay_output.jpg",
                      help="Output image path")
    p_ov.add_argument("--verbose", "-v", action="store_true")
    p_ov.set_defaults(func=_cmd_overlay)

    # --- build-dataset ---
    p_bd = sub.add_parser("build-dataset", help="Build YOLO dataset from backgrounds")
    p_bd.add_argument("--backgrounds", required=True,
                      help="Directory of background images")
    p_bd.add_argument("--output", default="datasets/synthetic_plates",
                      help="Output directory for the dataset")
    p_bd.add_argument("--count", type=int, default=1000,
                      help="Total images to generate")
    p_bd.add_argument("--plates-per-image", dest="plates_per_image", type=int, default=1,
                      help="Min plates per image")
    p_bd.add_argument("--max-plates", dest="max_plates", type=int, default=3,
                      help="Max plates per image")
    p_bd.add_argument("--val-split", dest="val_split", type=float, default=0.1,
                      help="Validation split fraction")
    p_bd.add_argument("--test-split", dest="test_split", type=float, default=0.05,
                      help="Test split fraction")
    p_bd.add_argument("--seed", type=int, default=None)
    p_bd.add_argument("--verbose", "-v", action="store_true")
    p_bd.set_defaults(func=_cmd_build_dataset)

    # --- grid ---
    p_gr = sub.add_parser("grid", help="Generate a preview grid of random plates")
    p_gr.add_argument("--type", default="all",
                      choices=["all", "mercosul", "old"],
                      help="Plate types to include")
    p_gr.add_argument("--count", type=int, default=20,
                      help="Number of plates in grid")
    p_gr.add_argument("--cols", type=int, default=5,
                      help="Grid columns")
    p_gr.add_argument("--output", default="plates_preview.jpg",
                      help="Output image path")
    p_gr.add_argument("--verbose", "-v", action="store_true")
    p_gr.set_defaults(func=_cmd_grid)

    # --- fonts ---
    p_fn = sub.add_parser("fonts", help="Show available fonts for plate rendering")
    p_fn.set_defaults(func=_cmd_fonts)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())