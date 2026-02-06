#!/usr/bin/env python3
"""
Face LoRA Trainer — LoRA Testing
==================================
Generate sample images using your trained LoRA to evaluate quality,
likeness, and flexibility across different prompts and scales.

Usage:
    python test_lora.py --lora output/face_lora.safetensors
    python test_lora.py --lora output/face_lora.safetensors --prompt "a portrait, studio lighting"
    python test_lora.py --lora output/face_lora.safetensors --scale 0.6 --num-images 8
    python test_lora.py --lora output/face_lora.safetensors --sweep  # test multiple scales
"""

import argparse
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionXLPipeline


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


# A set of varied test prompts to evaluate LoRA flexibility
DEFAULT_TEST_PROMPTS = [
    "a portrait, looking at the camera, neutral expression, studio lighting",
    "a portrait, slight smile, natural outdoor lighting",
    "a portrait, 3/4 view, dramatic side lighting",
    "a portrait, looking slightly away, soft diffused lighting",
]


def main():
    parser = argparse.ArgumentParser(
        description="Test your trained LoRA by generating sample images"
    )
    parser.add_argument(
        "--lora", type=str, required=True,
        help="Path to the trained LoRA .safetensors file"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Custom test prompt (trigger word is added automatically)"
    )
    parser.add_argument(
        "--negative-prompt", type=str,
        default="low quality, blurry, deformed, watermark, worst quality",
        help="Negative prompt"
    )
    parser.add_argument(
        "--scale", type=float, default=0.8,
        help="LoRA strength 0.0-1.0 (default: 0.8)"
    )
    parser.add_argument(
        "--num-images", type=int, default=4,
        help="Number of images to generate (default: 4)"
    )
    parser.add_argument(
        "--output", type=str, default="./test_output",
        help="Output directory for test images (default: ./test_output)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Inference steps (default: 30)"
    )
    parser.add_argument(
        "--guidance", type=float, default=7.5,
        help="Guidance scale (default: 7.5)"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Generate images at multiple LoRA scales (0.4, 0.6, 0.8, 1.0) "
             "to find the sweet spot"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Base model to use for inference (overrides config). "
             "Use this to test your LoRA with different base models "
             "(e.g., an Illustrious checkpoint path)."
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    trigger = config.get("dataset", {}).get("trigger_word", "ohwx")
    class_word = config.get("dataset", {}).get("class_word", "man")
    model_path = args.model or config.get("paths", {}).get(
        "pretrained_model", "stabilityai/stable-diffusion-xl-base-1.0"
    )

    # Validate LoRA path
    lora_path = Path(args.lora)
    if not lora_path.exists():
        print(f"Error: LoRA file not found: {lora_path}")
        print("Check your output directory for available checkpoints.")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine prompts and scales
    if args.sweep:
        scales = [0.4, 0.6, 0.8, 1.0]
        prompts = DEFAULT_TEST_PROMPTS[:2]  # Use 2 prompts for sweep
    else:
        scales = [args.scale]
        if args.prompt:
            prompts = [args.prompt]
        else:
            prompts = DEFAULT_TEST_PROMPTS[:args.num_images]
            # If more images than prompts, cycle through them
            while len(prompts) < args.num_images:
                prompts.extend(DEFAULT_TEST_PROMPTS)
            prompts = prompts[:args.num_images]

    # Add trigger word to prompts
    full_prompts = []
    for p in prompts:
        if trigger in p:
            full_prompts.append(p)
        else:
            full_prompts.append(f"{trigger}, a {class_word}, {p}")

    total_images = len(full_prompts) * len(scales)

    print("=" * 60)
    print("LoRA Test — Sample Generation")
    print("=" * 60)
    print(f"LoRA:         {lora_path}")
    print(f"Base model:   {model_path}")
    print(f"Trigger word: {trigger}")
    print(f"Scales:       {scales}")
    print(f"Images:       {total_images}")
    print(f"Output:       {output_dir}")
    print()

    # Load pipeline
    print("Loading SDXL pipeline...")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    except ValueError:
        # fp16 variant not available, try without
        print("  (fp16 variant not found, loading standard weights)")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
    pipe.enable_model_cpu_offload()

    print("Loading LoRA weights...")
    pipe.load_lora_weights(str(lora_path), adapter_name="face")
    print("Ready.\n")

    count = 0
    for scale in scales:
        # Apply LoRA at the current scale using set_adapters (cleaner than fuse/unfuse)
        pipe.set_adapters(["face"], adapter_weights=[scale])

        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        for i, prompt in enumerate(full_prompts):
            image = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
                width=1024,
                height=1024,
            ).images[0]

            filename = f"test_s{scale:.1f}_{i:02d}.png"
            out_path = output_dir / filename
            image.save(out_path)
            count += 1

            print(f"  [{count}/{total_images}] scale={scale:.1f} | {out_path.name}")
            print(f"    prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    # Cleanup
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone! {count} test images saved to {output_dir}")

    if args.sweep:
        print()
        print("Compare images across scales:")
        print("  - Low scale (0.4): More flexibility, less likeness")
        print("  - High scale (1.0): Strong likeness, less flexibility")
        print("  - Sweet spot is usually 0.6-0.8")


if __name__ == "__main__":
    main()
