#!/usr/bin/env python3
"""
Face LoRA Trainer — Regularisation Image Generator
====================================================
Generates class images using the SDXL base model to prevent the LoRA
from associating ALL instances of the class word with your face.

Without regularisation, prompting "a man" might always produce your face.
With regularisation, the model learns your face is specifically tied to
the trigger word while the class word retains its general meaning.

Usage:
    python generate_reg.py                        # use config.yaml defaults
    python generate_reg.py --num-images 100        # generate 100 images
    python generate_reg.py --config my_config.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionXLPipeline


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate regularisation images for LoRA training"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--num-images", type=int, default=None,
        help="Number of images to generate (overrides config)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for generation — increase if you have VRAM to spare"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    model_path = config["paths"]["pretrained_model"]
    reg_dir = Path(args.output or config["paths"]["reg_data_dir"])
    reg_config = config.get("reg_generation", {})
    class_word = config["dataset"]["class_word"]

    num_images = args.num_images or reg_config.get("num_images", 200)
    prompt = reg_config.get("prompt", f"a photo of a {class_word}, portrait")
    negative_prompt = reg_config.get("negative_prompt", "low quality, blurry, deformed")
    steps = reg_config.get("num_inference_steps", 30)
    guidance = reg_config.get("guidance_scale", 7.5)
    seed = config["training"].get("seed", 42)

    reg_dir.mkdir(parents=True, exist_ok=True)

    # Count existing reg images
    existing_images = sorted(reg_dir.glob("reg_*.png"))
    existing_count = len(existing_images)

    if existing_count >= num_images:
        print(f"Already have {existing_count} reg images (target: {num_images}).")
        print("Use --num-images with a higher number to generate more.")
        return

    remaining = num_images - existing_count
    start_idx = existing_count

    print("=" * 60)
    print("Regularisation Image Generator")
    print("=" * 60)
    print(f"Model:    {model_path}")
    print(f"Prompt:   {prompt}")
    print(f"Images:   {remaining} to generate ({existing_count} existing)")
    print(f"Output:   {reg_dir}")
    print(f"Steps:    {steps}")
    print(f"Guidance: {guidance}")
    print()

    # Load pipeline with CPU offloading for 12GB VRAM
    print("Loading SDXL pipeline (this may download the model on first run — ~6.5GB)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()

    print("Pipeline loaded. Generating images...\n")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    try:
        for i in range(remaining):
            idx = start_idx + i
            image_path = reg_dir / f"reg_{idx:04d}.png"
            caption_path = reg_dir / f"reg_{idx:04d}.txt"

            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                width=1024,
                height=1024,
            )

            image = result.images[0]
            image.save(image_path)

            # Caption file contains the prompt (without trigger word)
            caption_path.write_text(prompt, encoding="utf-8")

            progress = f"[{i + 1}/{remaining}]"
            print(f"  {progress} {image_path.name}")

    except KeyboardInterrupt:
        generated = i
        print(f"\n\nInterrupted after {generated} images.")
        print("Images generated so far are preserved. Run again to continue.")

    # Cleanup
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total = len(list(reg_dir.glob("reg_*.png")))
    print(f"\nDone! Total regularisation images: {total}")
    print(f"Saved to: {reg_dir}")
    print()
    print("Next step: python train.py")


if __name__ == "__main__":
    main()
