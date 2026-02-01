#!/usr/bin/env python3
"""
Face LoRA Trainer — Dataset Preparation
========================================
Processes raw photos for LoRA training: resizes, crops, and organises
images into the training directory.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --input ./my_photos --resolution 1024
    python prepare_dataset.py --no-crop  # maintain aspect ratio (uses bucketing)
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def process_image_crop(input_path: Path, output_path: Path, resolution: int) -> Path:
    """Resize smallest edge to resolution, then center-crop to square."""
    img = Image.open(input_path)
    img = ImageOps.exif_transpose(img)  # fix rotation from EXIF metadata

    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    scale = resolution / min(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop to square
    left = (new_w - resolution) // 2
    top = (new_h - resolution) // 2
    img = img.crop((left, top, left + resolution, top + resolution))

    output_file = output_path / f"{input_path.stem}.png"
    img.save(output_file, "PNG")
    return output_file


def process_image_no_crop(input_path: Path, output_path: Path, resolution: int) -> Path:
    """Resize longest edge to resolution, maintaining aspect ratio."""
    img = Image.open(input_path)
    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((resolution, resolution), Image.LANCZOS)

    output_file = output_path / f"{input_path.stem}.png"
    img.save(output_file, "PNG")
    return output_file


def get_image_files(directory: Path) -> list[Path]:
    """Find all supported image files in a directory."""
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset images for LoRA training"
    )
    parser.add_argument(
        "--input", type=str, default="./dataset/raw",
        help="Directory containing raw photos (default: ./dataset/raw)"
    )
    parser.add_argument(
        "--output", type=str, default="./dataset/train",
        help="Output directory for processed images (default: ./dataset/train)"
    )
    parser.add_argument(
        "--resolution", type=int, default=1024,
        help="Target resolution in pixels (default: 1024)"
    )
    parser.add_argument(
        "--no-crop", action="store_true",
        help="Resize without cropping — maintains aspect ratio. "
             "Use this if your photos have varying aspect ratios and you "
             "have bucket resolution enabled in training (enabled by default)."
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG quality if saving as JPEG (default: 95, unused for PNG)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Create it and add your photos, then run this script again.")
        sys.exit(1)

    images = get_image_files(input_dir)

    if not images:
        print(f"No images found in {input_dir}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "resize (no crop, aspect ratio preserved)" if args.no_crop else "resize + center crop"
    print(f"Processing {len(images)} images")
    print(f"Mode: {mode}")
    print(f"Target resolution: {args.resolution}x{args.resolution}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    success = 0
    failed = 0

    for i, img_path in enumerate(images, 1):
        try:
            if args.no_crop:
                out_file = process_image_no_crop(img_path, output_dir, args.resolution)
            else:
                out_file = process_image_crop(img_path, output_dir, args.resolution)

            out_img = Image.open(out_file)
            print(f"  [{i}/{len(images)}] {img_path.name} -> {out_file.name} ({out_img.size[0]}x{out_img.size[1]})")
            success += 1

        except Exception as e:
            print(f"  [{i}/{len(images)}] FAILED: {img_path.name} — {e}")
            failed += 1

    print(f"\nDone! {success} processed, {failed} failed.")
    print(f"Images saved to: {output_dir}")

    if success > 0:
        print(f"\nNext step: python caption_images.py")


if __name__ == "__main__":
    main()
