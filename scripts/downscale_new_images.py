#!/usr/bin/env python3
"""Downscale new image batches into the project data directory."""

import argparse
import logging
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: Path) -> Iterable[Path]:
    """Yield image paths below *root* that match SUPPORTED_SUFFIXES."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def downscale_image(src: Path, dst: Path, max_side: int, quality: int, overwrite: bool) -> bool:
    """Downscale *src* into *dst* and return True if an image was written."""
    if dst.exists() and not overwrite:
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)
        # Preserve aspect ratio while constraining the longest edge.
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        save_kwargs = {}
        if dst.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        elif dst.suffix.lower() == ".png":
            save_kwargs["optimize"] = True

        if image.mode in {"RGBA", "LA"} and dst.suffix.lower() in {".jpg", ".jpeg"}:
            image = image.convert("RGB")

        image.save(dst, **save_kwargs)

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="new_files",
        type=Path,
        help="Directory with newly delivered image batches (default: new_files).",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("claw_segmentation") / "data",
        type=Path,
        help="Target data directory (default: claw_segmentation/data).",
    )
    parser.add_argument(
        "--max-side",
        default=1024,
        type=int,
        help="Maximum size for the longer image edge (pixels, default: 1024).",
    )
    parser.add_argument(
        "--quality",
        default=90,
        type=int,
        help="JPEG quality when writing .jpg/.jpeg files (default: 90).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    written = 0
    skipped = 0

    for src in iter_images(input_dir):
        rel_path = src.relative_to(input_dir)
        dst = output_dir / rel_path

        if downscale_image(src, dst, args.max_side, args.quality, args.overwrite):
            written += 1
            logging.info("Saved %s -> %s", src, dst)
        else:
            skipped += 1

    print(
        "Processed images from",
        input_dir,
        "to",
        output_dir,
        "| written:",
        written,
        "| skipped:",
        skipped,
    )


if __name__ == "__main__":
    main()
