#!/usr/bin/env python3
"""
Face LoRA Trainer — Training Script
=====================================
Reads config.yaml, sets up the sd-scripts directory structure,
and launches SDXL LoRA training via Kohya sd-scripts.

Usage:
    python train.py                     # train with config.yaml
    python train.py --config my.yaml    # use a different config
    python train.py --dry-run           # print command without executing
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load and validate YAML configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_paths = ["pretrained_model", "sd_scripts_dir", "train_data_dir", "output_dir", "output_name"]
    for key in required_paths:
        if key not in config.get("paths", {}):
            print(f"Error: Missing required config field: paths.{key}")
            sys.exit(1)

    return config


def link_or_copy(src: Path, dest: Path):
    """Create symlink on Linux/Mac, copy on Windows."""
    if dest.exists():
        return
    if platform.system() == "Windows":
        shutil.copy2(str(src), str(dest))
    else:
        os.symlink(src.resolve(), dest)


def setup_training_dirs(config: dict) -> tuple[Path, Path | None]:
    """
    Create the sd-scripts dataset directory structure.

    sd-scripts expects:
        train_data_dir/{num_repeats}_{trigger_word class_word}/
            image1.png
            image1.txt
            ...
    """
    train_src = Path(config["paths"]["train_data_dir"])
    trigger = config["dataset"]["trigger_word"]
    class_word = config["dataset"]["class_word"]
    repeats = config["dataset"]["num_repeats"]

    if not train_src.exists():
        print(f"Error: Training data directory not found: {train_src}")
        print("Run prepare_dataset.py and caption_images.py first.")
        sys.exit(1)

    # Check for images
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    train_images = [f for f in train_src.iterdir() if f.suffix.lower() in image_exts]
    if not train_images:
        print(f"Error: No images found in {train_src}")
        sys.exit(1)

    # Check for captions
    captioned = [img for img in train_images if img.with_suffix(".txt").exists()]
    if not captioned:
        print(f"Error: No caption (.txt) files found in {train_src}")
        print("Run caption_images.py first.")
        sys.exit(1)
    if len(captioned) < len(train_images):
        print(f"Warning: {len(train_images) - len(captioned)} images have no captions.")
        print("Only captioned images will be used for training.")

    # Create structured directory for sd-scripts
    structured_dir = Path(config["paths"]["output_dir"]) / "_structured_dataset"
    train_structured = structured_dir / "train"

    # Clean previous structured dirs
    if structured_dir.exists():
        shutil.rmtree(str(structured_dir))

    # Create train subset folder: {repeats}_{trigger} {class_word}
    subset_name = f"{repeats}_{trigger} {class_word}"
    subset_dir = train_structured / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)

    # Link/copy training images and captions into the structured directory
    for f in train_src.iterdir():
        if f.is_file() and not f.name.startswith("."):
            link_or_copy(f, subset_dir / f.name)

    print(f"Training data: {len(captioned)} captioned images, {repeats}x repeats")
    print(f"  -> {len(captioned) * repeats} steps per epoch")

    # Handle regularisation images
    reg_structured = None
    reg_src_str = config["paths"].get("reg_data_dir", "")
    if reg_src_str:
        reg_src = Path(reg_src_str)
        reg_images = (
            [f for f in reg_src.iterdir() if f.suffix.lower() in image_exts]
            if reg_src.exists() else []
        )

        if reg_images:
            reg_repeats = config["dataset"].get("reg_repeats", 1)
            reg_structured = structured_dir / "reg"
            reg_subset_name = f"{reg_repeats}_{class_word}"
            reg_subset_dir = reg_structured / reg_subset_name
            reg_subset_dir.mkdir(parents=True, exist_ok=True)

            for f in reg_src.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    link_or_copy(f, reg_subset_dir / f.name)

            print(f"Reg data: {len(reg_images)} images, {reg_repeats}x repeats")
        else:
            print("Reg data: None (no images found, skipping regularisation)")
    else:
        print("Reg data: Disabled in config")

    return train_structured, reg_structured


def build_training_args(config: dict, train_dir: Path, reg_dir: Path | None) -> list[str]:
    """Build command-line arguments for sd-scripts sdxl_train_network.py."""
    args = []

    # ---- Model ----
    args.extend(["--pretrained_model_name_or_path", config["paths"]["pretrained_model"]])

    # ---- Dataset ----
    args.extend(["--train_data_dir", str(train_dir)])
    if reg_dir and reg_dir.exists():
        args.extend(["--reg_data_dir", str(reg_dir)])
    
    # Caption files use .txt extension
    args.extend(["--caption_extension", ".txt"])

    # ---- Output ----
    args.extend(["--output_dir", config["paths"]["output_dir"]])
    args.extend(["--output_name", config["paths"]["output_name"]])

    # ---- Network (LoRA) ----
    args.extend(["--network_module", config["network"]["module"]])
    args.extend(["--network_dim", str(config["network"]["rank"])])
    args.extend(["--network_alpha", str(config["network"]["alpha"])])

    # ---- Training ----
    t = config["training"]
    args.extend(["--resolution", str(t["resolution"])])
    args.extend(["--train_batch_size", str(t["batch_size"])])
    args.extend(["--max_train_epochs", str(t["epochs"])])
    args.extend(["--learning_rate", str(t["learning_rate"])])
    args.extend(["--lr_scheduler", t["scheduler"]])
    
    # Scheduler cycles (for cosine_with_restarts)
    if t["scheduler"] == "cosine_with_restarts":
        num_cycles = t.get("scheduler_num_cycles", 4)
        args.extend(["--lr_scheduler_num_cycles", str(num_cycles)])
    
    args.extend(["--seed", str(t["seed"])])
    args.extend(["--max_grad_norm", str(t["max_grad_norm"])])

    # Text encoder training
    # For SDXL, setting --text_encoder_lr enables text encoder training
    # The argument takes two values: one for each text encoder (CLIP-L and OpenCLIP-G)
    # NOTE: Cannot train text encoder while caching text encoder outputs
    train_te = t.get("train_text_encoder", False)
    cache_te = config["optimisation"].get("cache_text_encoder_outputs", False)
    
    if train_te and cache_te:
        print("  Note: Disabling text encoder training (incompatible with TE output caching)")
        print("        To train text encoder, set cache_text_encoder_outputs: false in config")
        train_te = False
    
    if train_te:
        te_lr = t.get("text_encoder_learning_rate", t["learning_rate"])
        # Pass the same LR for both text encoders
        args.extend(["--text_encoder_lr", str(te_lr), str(te_lr)])
    else:
        # Explicitly tell sd-scripts to only train UNet
        args.append("--network_train_unet_only")

    # Warmup
    warmup = t.get("warmup_ratio", 0)
    if warmup > 0:
        # sd-scripts interprets values < 1 as a ratio of total steps
        args.extend(["--lr_warmup_steps", str(warmup)])

    # Noise offset
    noise_offset = t.get("noise_offset", 0)
    if noise_offset and noise_offset > 0:
        args.extend(["--noise_offset", str(noise_offset)])

    # Min SNR gamma
    min_snr = t.get("min_snr_gamma", 0)
    if min_snr and min_snr > 0:
        args.extend(["--min_snr_gamma", str(min_snr)])

    # Optimizer
    optimizer = t["optimizer"]
    args.extend(["--optimizer_type", optimizer])

    # ---- VRAM Optimisation ----
    opt = config["optimisation"]
    args.extend(["--mixed_precision", opt["mixed_precision"]])

    if opt.get("gradient_checkpointing"):
        args.append("--gradient_checkpointing")

    if opt.get("cache_latents"):
        args.append("--cache_latents")
    if opt.get("cache_latents_to_disk"):
        args.append("--cache_latents_to_disk")

    if opt.get("cache_text_encoder_outputs"):
        args.append("--cache_text_encoder_outputs")
    if opt.get("cache_text_encoder_outputs_to_disk"):
        args.append("--cache_text_encoder_outputs_to_disk")

    if opt.get("xformers"):
        args.append("--xformers")
    elif opt.get("sdpa"):
        args.append("--sdpa")

    # ---- Saving ----
    sav = config["saving"]
    args.extend(["--save_every_n_epochs", str(sav["save_every_n_epochs"])])
    if sav.get("save_last_n_epochs", 0) > 0:
        args.extend(["--save_last_n_epochs", str(sav["save_last_n_epochs"])])
    args.extend(["--save_precision", sav["save_precision"]])
    args.extend(["--save_model_as", sav["save_model_as"]])
    
    # Save training state for resuming
    if sav.get("save_state"):
        args.append("--save_state")

    # ---- Additional flags ----
    # Enable aspect ratio bucketing (handles varying image sizes)
    args.append("--enable_bucket")

    # Caption shuffling (improves generalisation for tag-based captions)
    # shuffle_caption shuffles tags while keep_tokens preserves the first N tags (trigger word)
    ds = config.get("dataset", {})
    if ds.get("shuffle_caption", True):
        args.append("--shuffle_caption")
        keep_tokens = ds.get("keep_tokens", 1)
        if keep_tokens > 0:
            args.extend(["--keep_tokens", str(keep_tokens)])

    # Logging
    log_dir = str(Path(config["paths"]["output_dir"]) / "logs")
    args.extend(["--logging_dir", log_dir])

    return args


def find_accelerate_config() -> Path | None:
    """Find the accelerate config file shipped with this project."""
    project_config = Path(__file__).parent / "accelerate_config.yaml"
    if project_config.exists():
        return project_config
    return None


def main():
    parser = argparse.ArgumentParser(description="Face LoRA Trainer")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to training config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the full training command without executing it"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume training from a saved state directory (e.g., output/face_lora-000010-state)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Validate sd-scripts installation
    sd_scripts_dir = Path(config["paths"]["sd_scripts_dir"])
    train_script = sd_scripts_dir / "sdxl_train_network.py"

    if not train_script.exists():
        print("=" * 60)
        print("ERROR: Kohya sd-scripts not found!")
        print("=" * 60)
        print(f"Expected location: {sd_scripts_dir.resolve()}")
        print(f"Expected script:   {train_script}")
        print()
        print("Install sd-scripts:")
        print(f"  git clone https://github.com/kohya-ss/sd-scripts.git {sd_scripts_dir}")
        print(f"  cd {sd_scripts_dir}")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    # Setup directory structure
    print("=" * 60)
    print("Face LoRA Trainer")
    print("=" * 60)
    print()
    print("Setting up training directory structure...")
    train_dir, reg_dir = setup_training_dirs(config)

    # Build training arguments
    training_args = build_training_args(config, train_dir, reg_dir)

    # Handle resume from saved state
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"Error: Resume state not found: {resume_path}")
            print("Available states:")
            for state_dir in Path(config["paths"]["output_dir"]).glob("*-state"):
                print(f"  {state_dir}")
            sys.exit(1)
        training_args.extend(["--resume", str(resume_path)])
        print(f"Resuming from: {resume_path}")

    # Create output directory
    Path(config["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Find accelerate config
    accel_config = find_accelerate_config()

    # Build the full command
    cmd = [sys.executable, "-m", "accelerate.commands.launch"]

    if accel_config:
        cmd.extend(["--config_file", str(accel_config)])

    cmd.extend(["--num_cpu_threads_per_process", "1"])
    cmd.append(str(train_script.resolve()))  # Full path to script
    cmd.extend(training_args)

    # Set environment — add sd-scripts to PYTHONPATH so its modules can be found
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sd_scripts_dir.resolve()) + os.pathsep + env.get("PYTHONPATH", "")

    if args.dry_run:
        print()
        print("=" * 60)
        print("DRY RUN — Command that would be executed:")
        print("=" * 60)
        print()
        # Print as a readable multi-line command
        print(cmd[0])
        for part in cmd[1:]:
            if part.startswith("--"):
                print(f"  {part}", end="")
            else:
                print(f" {part}")
        print()
        print(f"PYTHONPATH includes: {sd_scripts_dir.resolve()}")
        return

    # Print summary
    print()
    print("Training Configuration:")
    print(f"  Base model:   {config['paths']['pretrained_model']}")
    print(f"  Train data:   {train_dir}")
    print(f"  Reg data:     {reg_dir or 'None'}")
    print(f"  Output:       {config['paths']['output_dir']}")
    print(f"  LoRA name:    {config['paths']['output_name']}")
    print(f"  LoRA rank:    {config['network']['rank']}")
    print(f"  LoRA alpha:   {config['network']['alpha']}")
    print(f"  Epochs:       {config['training']['epochs']}")
    print(f"  Resolution:   {config['training']['resolution']}")
    print(f"  Batch size:   {config['training']['batch_size']}")
    print(f"  LR:           {config['training']['learning_rate']}")
    print(f"  Optimizer:    {config['training']['optimizer']}")
    print(f"  Mixed prec:   {config['optimisation']['mixed_precision']}")
    print()
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    print()

    # Launch training (runs from current directory, not sd-scripts)
    result = subprocess.run(cmd, env=env)

    if result.returncode == 0:
        print()
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Checkpoints saved to: {config['paths']['output_dir']}")
        print()
        print("Next step: python test_lora.py --lora output/face_lora.safetensors")
    else:
        print()
        print("=" * 60)
        print(f"Training Failed (exit code {result.returncode})")
        print("=" * 60)
        print("Check the error messages above for details.")
        print("Common issues:")
        print("  - Out of VRAM: reduce network rank or batch size")
        print("  - Module not found: ensure sd-scripts dependencies are installed")
        print("  - CUDA error: check PyTorch and CUDA compatibility")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
