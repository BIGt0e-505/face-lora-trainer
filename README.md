# Face LoRA Trainer

A Python workflow for training SDXL LoRA models of your face, optimised for 12GB VRAM GPUs. Train a style-agnostic face LoRA against the SDXL 1.0 base model, then apply any style at inference time using style-specific checkpoints like [Illustrious](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0), [AnimagineXL](https://huggingface.co/cagliostrolab/animagine-xl-3.1), [Pony Diffusion](https://civitai.com/models/257749), or any other SDXL-based model.

## How It Works

This project trains a LoRA (Low-Rank Adaptation) that encodes your facial identity into a small, portable weights file. By training against the vanilla SDXL 1.0 base model rather than a stylised checkpoint, the resulting LoRA remains **style-agnostic** — it captures *who you are*, not *what style you're drawn in*. At inference time, you load your face LoRA into any SDXL-based model and the style comes from that model, not from your LoRA.

The workflow is fully Python-native and runs locally. No GUI, no cloud, no Colab.

## Features

- **Complete pipeline** from raw photos to trained LoRA: prep → caption → regularise → train → test
- **YAML configuration** for all settings — reproducible and version-controllable
- **12GB VRAM optimised** with latent caching, text encoder caching, gradient checkpointing, and 8-bit optimisers
- **Auto-captioning** with BLIP (with interactive review mode)
- **Regularisation image generation** using the SDXL base model
- **LoRA testing utility** with scale sweep for finding the sweet spot
- **Cross-platform** — works on Windows and Linux

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 12GB    | 16GB+       |
| System RAM| 32GB    | 64GB        |
| Disk Space| 20GB    | 40GB        |

The VRAM optimisations in `config.yaml` are tuned for 12GB GPUs (e.g., RTX 4070, RTX 3060 12GB, RTX PRO 3000). If you have more VRAM, you can disable some caching for faster training.

### Software

- Python 3.10 or later
- CUDA 12.x (or 11.8+ with appropriate PyTorch build)
- Git

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/face-lora-trainer.git
cd face-lora-trainer
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install PyTorch

Install the version matching your CUDA version. Check https://pytorch.org/get-started/locally/ for the correct command.

```bash
# Example for CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install project dependencies

```bash
pip install -r requirements.txt
```

### 5. Clone and install Kohya sd-scripts

```bash
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts
pip install -r requirements.txt
cd ..
```

> **Note:** The `config.yaml` expects sd-scripts at `./sd-scripts`. If you clone it elsewhere, update `paths.sd_scripts_dir` in the config.

### 6. Verify the setup

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import accelerate; print(f'Accelerate {accelerate.__version__}')"
```

Both should report their versions without errors. CUDA should show as available.

---

## Folder Structure

```
face-lora-trainer/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── config.yaml                # Training configuration (edit this)
├── accelerate_config.yaml     # Accelerate launcher config
├── .gitignore
│
├── prepare_dataset.py         # Step 1: Process raw photos
├── caption_images.py          # Step 2: Generate captions
├── generate_reg.py            # Step 3: Generate regularisation images
├── train.py                   # Step 4: Launch training
├── test_lora.py               # Step 5: Test your LoRA
│
├── dataset/
│   ├── raw/                   # Put your raw photos here
│   ├── train/                 # Processed images + captions (auto-generated)
│   └── reg/                   # Regularisation images (auto-generated)
│
├── output/                    # Training checkpoints (auto-generated)
│
└── sd-scripts/                # Kohya sd-scripts (cloned separately)
```

---

## Quick Start

```bash
# 1. Put 15-30 photos in dataset/raw/

# 2. Process photos
python prepare_dataset.py

# 3. Generate captions
python caption_images.py --interactive

# 4. Generate regularisation images (optional but recommended)
python generate_reg.py

# 5. Train
python train.py

# 6. Test
python test_lora.py --lora output/face_lora.safetensors --sweep
```

---

## Detailed Workflow

### Step 1: Preparing Your Photos

Place your photos in `dataset/raw/`. The quality and variety of your training data is the single biggest factor in LoRA quality.

#### What to include

Aim for **15–30 photos**. More is fine (up to ~50), but quality and variety matter more than quantity.

| Aspect | What You Want | Why |
|--------|---------------|-----|
| **Angles** | Front, 3/4, profile, slight up/down | The LoRA needs to learn your face from multiple viewpoints |
| **Lighting** | Natural, indoor, soft, hard, different directions | Prevents the LoRA from baking in a single lighting style |
| **Expressions** | Neutral, smiling, serious, talking | Adds flexibility at inference time |
| **Distance** | Close-up face, head and shoulders, upper body | Teaches the model your face at different scales |
| **Background** | Varied or neutral | Prevents background association |

#### What to avoid

- **Sunglasses or face-obscuring accessories** — the model can't learn what it can't see
- **Heavy filters or colour grading** — bakes artificial colour into the LoRA
- **Group photos** — the model may confuse which face is yours
- **Very low resolution** — aim for at least 512×512 before processing
- **Inconsistent appearance** — keep the same hairstyle, facial hair, and glasses throughout unless you want the LoRA to generalise across those variations

#### Photo sources

Phone selfies work well. If you're gathering existing photos, try to find ones with good focus on your face. Screenshots from video calls are usually too low quality.

### Step 2: Processing Images

```bash
python prepare_dataset.py
```

This resizes and crops your photos to 1024×1024 (matching SDXL's native resolution) and saves them as PNG to `dataset/train/`.

**Options:**

```bash
# Custom directories
python prepare_dataset.py --input ./my_photos --output ./dataset/train

# Different resolution (not recommended unless you have a reason)
python prepare_dataset.py --resolution 768

# Maintain aspect ratio instead of cropping to square
# (works with bucket resolution in training, enabled by default)
python prepare_dataset.py --no-crop
```

> **Tip:** If your photos are mostly portrait orientation (taller than wide), `--no-crop` can preserve more of the image. Bucket resolution in training handles varying aspect ratios.

### Step 3: Captioning / Tagging

```bash
python caption_images.py --interactive
```

This generates a `.txt` caption file for each image using the BLIP vision-language model, with your trigger word prepended. The `--interactive` flag lets you review and edit each caption before saving.

**How captioning works:**

Each image gets a corresponding `.txt` file with the same name:
```
dataset/train/
├── photo_01.png
├── photo_01.txt    # "ohwx, a photo of a person facing the camera, short dark hair..."
├── photo_02.png
├── photo_02.txt    # "ohwx, a person looking slightly to the right, wearing a grey shirt..."
```

**The trigger word** (default: `ohwx`) is a unique token that the model learns to associate with your face. At inference time, including the trigger word in your prompt activates the LoRA's learned identity.

**What makes a good caption:**

- Describe what's visible: angle, expression, clothing, lighting, background
- Don't include your name — the trigger word replaces your identity
- Don't include style descriptors (no "anime", "realistic", "photo") — keep it factual
- Keep captions concise but descriptive, 10–30 words is ideal

**Example captions:**

```
ohwx, a person facing the camera, short brown hair, wearing a navy t-shirt, neutral expression, indoor lighting
ohwx, a person in 3/4 view, slight smile, outdoor natural lighting, blurred green background
ohwx, a person looking slightly upward, glasses, stubble, warm side lighting
```

**Options:**

```bash
# Fully automatic (no review)
python caption_images.py

# Write all captions manually (no BLIP)
python caption_images.py --manual

# Replace existing captions
python caption_images.py --overwrite --interactive

# Custom trigger word
python caption_images.py --trigger-word mysecrettoken
```

> **Important:** After auto-captioning, review the `.txt` files in `dataset/train/` and fix any mistakes. BLIP sometimes hallucinates details or gets descriptions wrong. Accurate captions directly impact training quality.

### Step 4: Regularisation Images (Optional but Recommended)

```bash
python generate_reg.py
```

This generates ~200 generic images of the class word (e.g., "a man") using the SDXL base model. These images teach the LoRA that the class word refers to people in general, not specifically to you.

**Without regularisation:** Prompting "a man" after training might always produce your face.
**With regularisation:** Only the trigger word (`ohwx`) produces your face. "A man" produces generic faces as expected.

The script saves images and captions to `dataset/reg/`. It uses CPU offloading to fit within 12GB VRAM and can resume if interrupted.

**Options:**

```bash
# Generate fewer images (faster, slightly less effective)
python generate_reg.py --num-images 100

# Custom output directory
python generate_reg.py --output ./my_reg_images
```

> **Note:** This step downloads the full SDXL model (~6.5GB) on first run. The same model is used for training, so this is a one-time download.

### Step 5: Configuration

Edit `config.yaml` before training. The defaults are tuned for 12GB VRAM and should work well out of the box. Key settings you may want to adjust:

**Must check:**
- `dataset.trigger_word` — your chosen trigger word (must match what you used in captioning)
- `dataset.class_word` — "man", "woman", or "person"
- `paths.pretrained_model` — leave as SDXL 1.0 for style-agnostic training

**May want to adjust:**
- `network.rank` — 16 (lightweight) to 64 (maximum detail). 32 is a good default.
- `training.epochs` — 10–20. More epochs = more likeness but higher overfitting risk.
- `training.learning_rate` — 1e-4 is a safe default.

**Advanced (usually leave alone):**
- `optimisation.*` — VRAM settings. Only change if you have more or less VRAM.
- `training.noise_offset` — Helps with contrast. 0.0357 is a tested default.
- `training.min_snr_gamma` — Training stability. 5 is standard.

See the comments in `config.yaml` for detailed explanations of every parameter.

### Step 6: Training

```bash
python train.py
```

The script:
1. Reads `config.yaml`
2. Creates the folder structure that sd-scripts expects
3. Launches training via `accelerate`

**What to expect:**
- First run downloads the SDXL model if not already cached (~6.5GB)
- Latent and text encoder caching runs once at the start (a few minutes)
- Actual training: ~1–3 hours depending on dataset size and epoch count
- Checkpoints saved every 5 epochs (configurable) to `output/`

**Options:**

```bash
# Preview the training command without running it
python train.py --dry-run

# Use a different config
python train.py --config experiment_02.yaml
```

**Monitoring:** Training progress is printed to the console. For TensorBoard logging, the logs are saved to `output/logs/` — you can view them with:

```bash
tensorboard --logdir output/logs
```

### Step 7: Testing Your LoRA

```bash
python test_lora.py --lora output/face_lora.safetensors
```

This generates sample images using your trained LoRA so you can evaluate quality.

**Scale sweep** — find the optimal LoRA strength:

```bash
python test_lora.py --lora output/face_lora.safetensors --sweep
```

This generates images at scales 0.4, 0.6, 0.8, and 1.0. Compare them:
- **Low scale (0.4):** More flexibility, less likeness
- **High scale (1.0):** Strong likeness, may be rigid
- **Sweet spot:** Usually 0.6–0.8

**Test intermediate checkpoints:**

```bash
# If you saved every 5 epochs, test each one
python test_lora.py --lora output/face_lora-000005.safetensors
python test_lora.py --lora output/face_lora-000010.safetensors
python test_lora.py --lora output/face_lora.safetensors  # final
```

Earlier checkpoints may have less likeness but more flexibility. Later ones may overfit.

**Custom prompts:**

```bash
python test_lora.py --lora output/face_lora.safetensors \
    --prompt "portrait, dramatic lighting, looking away" \
    --num-images 8
```

### Step 8: Using Your LoRA at Inference

Once you have a trained LoRA you're happy with, you can use it with any SDXL-based model.

**With your existing pipeline (diffusers):**

```python
from diffusers import StableDiffusionXLPipeline
import torch

# Load your preferred SDXL model (e.g., Illustrious)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "your-illustrious-model-path",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.enable_model_cpu_offload()

# Load your face LoRA
pipe.load_lora_weights("output/face_lora.safetensors")
pipe.fuse_lora(lora_scale=0.7)

# Generate — include trigger word in your prompt
image = pipe(
    prompt="ohwx, 1boy, anime style, portrait, detailed face",
    negative_prompt="low quality, worst quality",
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
```

**With ComfyUI or Automatic1111:**

Copy the `.safetensors` file to your UI's LoRA directory and activate it as you would any other LoRA. Use your trigger word in the prompt.

**Combining with style LoRAs:**

You can stack multiple LoRAs. Load your face LoRA alongside an Illustrious-compatible style LoRA:

```python
# Load face LoRA
pipe.load_lora_weights("output/face_lora.safetensors", adapter_name="face")
# Load style LoRA
pipe.load_lora_weights("path/to/style_lora.safetensors", adapter_name="style")

# Set weights for each
pipe.set_adapters(["face", "style"], adapter_weights=[0.7, 0.6])
```

**With your dancing character video pipeline:**

Integrate the face LoRA loading into your existing Stable Diffusion XL + pose control workflow. The LoRA loads as a drop-in addition without changing the rest of your pipeline.

---

## Troubleshooting

### CUDA out of memory

If training crashes with an OOM error:
1. Ensure all caching options are enabled in `config.yaml` (they are by default)
2. Reduce `network.rank` from 32 to 16
3. Ensure `training.batch_size` is 1
4. Ensure `optimisation.gradient_checkpointing` is true
5. Try `mixed_precision: fp16` instead of `bf16`

### sd-scripts module not found

Make sure you've installed sd-scripts dependencies:

```bash
cd sd-scripts
pip install -r requirements.txt
cd ..
```

### bitsandbytes errors on Windows

If you get errors with the 8-bit optimizer on Windows:

```bash
pip install bitsandbytes --upgrade
```

bitsandbytes 0.43.0+ has native Windows support. If issues persist, change `training.optimizer` to `AdamW` in config (uses more VRAM but avoids bitsandbytes).

### Overfitting (every output looks like a training photo)

- Reduce `training.epochs` (try 8–10)
- Reduce `network.rank` (try 16)
- Add more varied training images
- Ensure regularisation images are being used

### Poor likeness (doesn't look like you)

- Increase `training.epochs` (try 18–20)
- Increase `network.rank` (try 48 or 64, VRAM permitting)
- Review captions — ensure trigger word is present and descriptions are accurate
- Add more training images with clear, well-lit face shots
- Check that `training.train_text_encoder` is true

### Accelerate config issues

If `accelerate` complains about missing config, run the interactive setup:

```bash
accelerate config
```

Or use the provided `accelerate_config.yaml` — `train.py` uses it automatically.

### Model download slow or failing

The SDXL 1.0 model downloads from HuggingFace on first run. If it's slow:

```bash
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 python generate_reg.py
```

Or download it manually:

```bash
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir ./models/sdxl-base
```

Then update `paths.pretrained_model` in config to `./models/sdxl-base`.

---

## Configuration Reference

See `config.yaml` for the complete configuration with inline documentation. Here is a summary of the key sections:

| Section | Key Settings | Notes |
|---------|-------------|-------|
| `paths` | `pretrained_model`, `sd_scripts_dir`, data dirs | Use SDXL 1.0 HuggingFace ID for auto-download |
| `dataset` | `trigger_word`, `class_word`, `num_repeats` | Repeats × images = steps per epoch |
| `network` | `rank`, `alpha` | Rank 32 / alpha 16 is a good starting point |
| `training` | `epochs`, `learning_rate`, `optimizer` | AdamW8bit at 1e-4 for 15 epochs |
| `optimisation` | caching, precision, attention | All caching on for 12GB VRAM |
| `saving` | checkpoint frequency, format | safetensors, every 5 epochs |

---

## Tips

- **Start small:** Train with rank 16 and 10 epochs first. If likeness is good, you're done. If not, increase rank and epochs.
- **Save checkpoints frequently:** Set `save_every_n_epochs: 3` or even 2 during your first training run so you can compare stages.
- **Review captions carefully:** This is the highest-impact thing you can do. Bad captions = bad LoRA.
- **Use regularisation:** It takes extra time to generate the images, but the results are noticeably better.
- **Test with the base model first:** Before testing with Illustrious or other stylised checkpoints, test with the same SDXL 1.0 model you trained on. If it doesn't work there, it won't work elsewhere.
- **Keep your trigger word consistent:** Use the same trigger word across all your LoRA projects to build muscle memory.

---

## License

MIT License. Do what you like with the code.

The training scripts use [Kohya sd-scripts](https://github.com/kohya-ss/sd-scripts) (Apache 2.0) and [Stability AI's SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (CreativeML Open RAIL++-M License).

## Acknowledgements

- [Kohya ss](https://github.com/kohya-ss/sd-scripts) for the excellent training framework
- [Stability AI](https://stability.ai/) for SDXL
- [Salesforce](https://huggingface.co/Salesforce/blip-image-captioning-large) for the BLIP captioning model
