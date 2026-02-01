#!/usr/bin/env python3
"""
Face LoRA Trainer — Image Captioning
======================================
Generates descriptive captions for training images using BLIP.
Creates a .txt file alongside each image containing the caption
with your trigger word prepended.

Usage:
    python caption_images.py                          # auto-caption with defaults
    python caption_images.py --interactive             # review each caption
    python caption_images.py --trigger-word sks        # override trigger word
    python caption_images.py --overwrite               # replace existing captions
    python caption_images.py --input ./my_images       # custom input directory
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def load_blip_model(model_name: str, device: str):
    """Load BLIP captioning model."""
    print(f"Loading captioning model: {model_name}")
    print("(This may download the model on first run — ~1GB)")

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    print("Model loaded.\n")
    return processor, model


def generate_caption(
    image_path: Path,
    processor,
    model,
    device: str,
    prefix: str = "a photo of",
    max_length: int = 75,
) -> str:
    """Generate a natural language caption for a single image."""
    img = Image.open(image_path).convert("RGB")

    dtype = torch.float16 if device == "cuda" else torch.float32
    inputs = processor(img, text=prefix, return_tensors="pt").to(device, dtype)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=3,
            repetition_penalty=1.5,
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()


def get_image_files(directory: Path) -> list[Path]:
    """Find all supported image files in a directory."""
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions for training images using BLIP"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Image directory (overrides config)"
    )
    parser.add_argument(
        "--trigger-word", type=str, default=None,
        help="Trigger word to prepend (overrides config)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda or cpu (auto-detected if not set)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Review and optionally edit each caption before saving"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing caption files"
    )
    parser.add_argument(
        "--caption-prefix", type=str, default="a photo of a person",
        help="Prefix hint for BLIP captioning (default: 'a photo of a person')"
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Skip BLIP entirely — write all captions manually"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Resolve settings (CLI flags override config)
    image_dir = (
        Path(args.input) if args.input
        else Path(config.get("paths", {}).get("train_data_dir", "./dataset/train"))
    )
    trigger_word = (
        args.trigger_word
        or config.get("dataset", {}).get("trigger_word", "ohwx")
    )
    model_name = config.get("captioning", {}).get(
        "model", "Salesforce/blip-image-captioning-large"
    )
    append_tags = config.get("captioning", {}).get("append_tags", "")

    # Validate
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        print("Run prepare_dataset.py first, or specify --input.")
        sys.exit(1)

    images = get_image_files(image_dir)
    if not images:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    # Filter out already-captioned images unless overwriting
    if not args.overwrite:
        uncaptioned = [img for img in images if not img.with_suffix(".txt").exists()]
        already_done = len(images) - len(uncaptioned)
        if already_done > 0:
            print(f"Found {already_done} existing caption files (use --overwrite to replace).")
        if not uncaptioned:
            print("All images already have captions. Nothing to do.")
            return
        images = uncaptioned

    print(f"Captioning {len(images)} images")
    print(f"Trigger word: {trigger_word}")
    print(f"Directory: {image_dir}")
    print()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load BLIP model (unless manual mode)
    processor, model = None, None
    if not args.manual:
        processor, model = load_blip_model(model_name, device)

    for i, img_path in enumerate(images, 1):
        try:
            if args.manual:
                # Manual mode — prompt user for each caption
                print(f"\n[{i}/{len(images)}] {img_path.name}")
                caption_text = input("  Enter caption (describe what you see): ").strip()
                if not caption_text:
                    print("  Skipped (no caption entered).")
                    continue
            else:
                # Auto-generate with BLIP
                caption_text = generate_caption(
                    img_path, processor, model, device,
                    prefix=args.caption_prefix,
                )

            # Build final caption: trigger_word, generated caption, optional tags
            parts = [trigger_word, caption_text]
            if append_tags:
                parts.append(append_tags)
            final_caption = ", ".join(parts)

            # Interactive review
            if args.interactive and not args.manual:
                print(f"\n[{i}/{len(images)}] {img_path.name}")
                print(f"  Generated: {final_caption}")
                edit = input("  Press Enter to accept, or type replacement: ").strip()
                if edit:
                    final_caption = edit
                    # Ensure trigger word is still present
                    if trigger_word not in final_caption:
                        final_caption = f"{trigger_word}, {final_caption}"

            # Save caption file
            caption_path = img_path.with_suffix(".txt")
            caption_path.write_text(final_caption, encoding="utf-8")
            print(f"  [{i}/{len(images)}] {img_path.name} -> {final_caption}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Captions saved so far are preserved.")
            break
        except Exception as e:
            print(f"  [{i}/{len(images)}] FAILED: {img_path.name} — {e}")

    # Cleanup
    if model is not None:
        del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone! Caption files saved alongside images in {image_dir}")
    print()
    print("IMPORTANT: Review the .txt files and refine captions if needed.")
    print("Good captions describe what's visible without naming you.")
    print("The trigger word replaces your identity.")
    print()
    print("Example caption:")
    print(f'  {trigger_word}, a person facing the camera, short dark hair, '
          f'wearing a blue shirt, neutral expression, indoor lighting')
    print()
    print("Next step: python generate_reg.py  (optional but recommended)")
    print("      or:  python train.py")


if __name__ == "__main__":
    main()
