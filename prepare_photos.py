#!/usr/bin/env python3
"""
Face LoRA Trainer — Photo Preparation with Head Detection
===========================================================
Processes camera photos for LoRA training by detecting the head/face
and centering the crop on it. Uses MediaPipe for fast, accurate face
detection that works well across angles and lighting conditions.

Usage:
    python prepare_photos.py --input ./camera_photos --output ./dataset/train
    python prepare_photos.py --input ./photos --output ./dataset/train --resolution 1024
    python prepare_photos.py --input ./photos --output ./dataset/train --padding 0.5
    python prepare_photos.py --input ./photos --output ./dataset/train --no-crop
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import mediapipe as mp
except ImportError:
    print("Error: mediapipe not installed.")
    print("Install it with: pip install mediapipe")
    sys.exit(1)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif"}


class HeadDetector:
    """Face/head detection using MediaPipe."""
    
    def __init__(self, min_confidence: float = 0.5):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,  # 1 = full range model (better for varied distances)
            min_detection_confidence=min_confidence,
        )
    
    def detect(self, image: np.ndarray) -> tuple[int, int, int, int] | None:
        """
        Detect face and return bounding box (x, y, width, height).
        Returns None if no face detected.
        """
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        
        if not results.detections:
            return None
        
        # Take the first (most confident) detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w = image.shape[:2]
        
        # Convert relative coords to absolute pixels
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        return (x, y, width, height)
    
    def close(self):
        self.detector.close()


def expand_bbox_for_head(
    bbox: tuple[int, int, int, int],
    img_width: int,
    img_height: int,
    padding: float = 0.6,
) -> tuple[int, int, int, int]:
    """
    Expand face bounding box to include full head, hair, and some shoulders.
    
    MediaPipe returns a tight face box. We expand it to capture:
    - Hair/top of head (expand up)
    - Ears (expand sides)
    - Neck/upper shoulders (expand down slightly)
    
    Args:
        bbox: (x, y, width, height) of detected face
        img_width, img_height: image dimensions
        padding: how much to expand (0.5 = 50% larger on each side)
    
    Returns:
        Expanded (x, y, width, height) clamped to image bounds
    """
    x, y, w, h = bbox
    
    # Calculate center of face
    cx = x + w // 2
    cy = y + h // 2
    
    # Expand the box
    # More expansion upward for hair, less downward
    expand_up = int(h * (padding + 0.3))  # Extra for hair
    expand_down = int(h * (padding - 0.1))  # Less for chin/neck
    expand_sides = int(w * padding)
    
    new_x = max(0, x - expand_sides)
    new_y = max(0, y - expand_up)
    new_w = min(img_width - new_x, w + 2 * expand_sides)
    new_h = min(img_height - new_y, h + expand_up + expand_down)
    
    return (new_x, new_y, new_w, new_h)


def make_square_crop(
    bbox: tuple[int, int, int, int],
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert a rectangular bbox to a square crop centered on the same point.
    Ensures the square fits within image bounds.
    """
    x, y, w, h = bbox
    
    # Center point
    cx = x + w // 2
    cy = y + h // 2
    
    # Square side = max of width and height
    side = max(w, h)
    
    # Initial square bounds
    new_x = cx - side // 2
    new_y = cy - side // 2
    
    # Clamp to image bounds, shifting if needed
    if new_x < 0:
        new_x = 0
    if new_y < 0:
        new_y = 0
    if new_x + side > img_width:
        new_x = img_width - side
    if new_y + side > img_height:
        new_y = img_height - side
    
    # Final clamp (in case image is smaller than requested square)
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    side = min(side, img_width - new_x, img_height - new_y)
    
    return (new_x, new_y, side, side)


def process_image_with_detection(
    input_path: Path,
    output_path: Path,
    detector: HeadDetector,
    resolution: int,
    padding: float,
    allow_fallback: bool = True,
) -> tuple[bool, str]:
    """
    Process a single image: detect head, crop, resize, save.
    
    Returns:
        (success: bool, message: str)
    """
    # Load with OpenCV for MediaPipe compatibility
    img_cv = cv2.imread(str(input_path))
    if img_cv is None:
        return False, "Failed to load image"
    
    img_height, img_width = img_cv.shape[:2]
    
    # Detect face
    face_bbox = detector.detect(img_cv)
    
    if face_bbox is None:
        if allow_fallback:
            # Fallback: center crop
            min_dim = min(img_width, img_height)
            x = (img_width - min_dim) // 2
            y = (img_height - min_dim) // 2
            crop_bbox = (x, y, min_dim, min_dim)
            method = "center-crop (no face detected)"
        else:
            return False, "No face detected"
    else:
        # Expand face box to head box
        head_bbox = expand_bbox_for_head(face_bbox, img_width, img_height, padding)
        
        # Make it square
        crop_bbox = make_square_crop(head_bbox, img_width, img_height)
        method = "head-centered"
    
    # Crop
    x, y, w, h = crop_bbox
    cropped = img_cv[y:y+h, x:x+w]
    
    # Convert to PIL for high-quality resize
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(cropped_rgb)
    
    # Resize to target resolution
    img_pil = img_pil.resize((resolution, resolution), Image.LANCZOS)
    
    # Save
    output_file = output_path / f"{input_path.stem}.png"
    img_pil.save(output_file, "PNG")
    
    return True, f"{method} -> {output_file.name}"


def process_image_no_crop(
    input_path: Path,
    output_path: Path,
    detector: HeadDetector,
    resolution: int,
) -> tuple[bool, str]:
    """
    Process without cropping to square — just resize maintaining aspect ratio.
    Still detects face to verify there is one in the image.
    """
    # Load with PIL
    img = Image.open(input_path)
    img = ImageOps.exif_transpose(img)
    
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Check for face using OpenCV/MediaPipe
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    face_bbox = detector.detect(img_cv)
    
    if face_bbox is None:
        return False, "No face detected (skipped)"
    
    # Resize longest edge to resolution
    img.thumbnail((resolution, resolution), Image.LANCZOS)
    
    output_file = output_path / f"{input_path.stem}.png"
    img.save(output_file, "PNG")
    
    return True, f"resized ({img.size[0]}x{img.size[1]}) -> {output_file.name}"


def get_image_files(directory: Path) -> list[Path]:
    """Find all supported image files in a directory."""
    files = []
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare camera photos for LoRA training with head detection"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input directory containing camera photos"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./dataset/train",
        help="Output directory for processed images (default: ./dataset/train)"
    )
    parser.add_argument(
        "--resolution", "-r", type=int, default=1024,
        help="Target resolution in pixels (default: 1024)"
    )
    parser.add_argument(
        "--padding", "-p", type=float, default=0.6,
        help="Padding around detected face (0.0-1.0, default: 0.6). "
             "Higher values include more of the head/shoulders."
    )
    parser.add_argument(
        "--no-crop", action="store_true",
        help="Don't crop to square — just resize maintaining aspect ratio. "
             "Use with bucket resolution in training."
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Minimum face detection confidence (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--skip-no-face", action="store_true",
        help="Skip images where no face is detected instead of falling back to center crop"
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    images = get_image_files(input_dir)
    
    if not images:
        print(f"No images found in {input_dir}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Photo Preparation with Head Detection")
    print("=" * 60)
    print(f"Input:      {input_dir} ({len(images)} images)")
    print(f"Output:     {output_dir}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Padding:    {args.padding}")
    print(f"Mode:       {'resize only' if args.no_crop else 'head-centered square crop'}")
    print()
    
    # Initialize detector
    detector = HeadDetector(min_confidence=args.confidence)
    
    success = 0
    failed = 0
    no_face = 0
    
    for i, img_path in enumerate(images, 1):
        try:
            if args.no_crop:
                ok, msg = process_image_no_crop(
                    img_path, output_dir, detector, args.resolution
                )
            else:
                ok, msg = process_image_with_detection(
                    img_path, output_dir, detector,
                    args.resolution, args.padding,
                    allow_fallback=not args.skip_no_face,
                )
            
            status = "✓" if ok else "✗"
            print(f"  [{i}/{len(images)}] {status} {img_path.name}: {msg}")
            
            if ok:
                success += 1
            elif "no face" in msg.lower():
                no_face += 1
                failed += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"  [{i}/{len(images)}] ✗ {img_path.name}: ERROR - {e}")
            failed += 1
    
    detector.close()
    
    print()
    print("=" * 60)
    print(f"Done! {success} processed, {failed} failed")
    if no_face > 0:
        print(f"  ({no_face} images had no detectable face)")
    print(f"Output saved to: {output_dir}")
    
    if success > 0:
        print()
        print("Next step: python caption_images.py --interactive")


if __name__ == "__main__":
    main()
