"""Inference pipeline: image -> segmentation mask -> polygon JSON."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from config import INPUT_SIZE, NUM_CLASSES
from vectorize import mask_to_polygons


def get_device() -> torch.device:
    """Select best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def preprocess_image(image_path: str) -> tuple[torch.Tensor, dict]:
    """Load and preprocess an image for model inference.

    Resizes preserving aspect ratio to fit INPUT_SIZE, pads to
    INPUT_SIZE x INPUT_SIZE with white, and converts to a float tensor.

    Args:
        image_path: Path to the input image file.

    Returns:
        Tuple of (tensor, meta) where:
          - tensor: float32 tensor of shape (1, 3, INPUT_SIZE, INPUT_SIZE)
            with values in [0, 1]
          - meta: dict with keys:
              original_size: (height, width) of the original image
              padding: (pad_top, pad_left, content_h, content_w) describing
                  how the image was placed into the padded canvas
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size  # PIL uses (width, height)

    # Compute new size preserving aspect ratio to fit within INPUT_SIZE
    scale = INPUT_SIZE / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize with high-quality resampling
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create white canvas and paste resized image centered (matching training)
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (255, 255, 255))
    pad_left = (INPUT_SIZE - new_w) // 2
    pad_top = (INPUT_SIZE - new_h) // 2
    canvas.paste(img_resized, (pad_left, pad_top))

    # Convert to float tensor: (3, H, W) in [0, 1]
    arr = np.array(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    meta = {
        "original_size": (orig_h, orig_w),
        "padding": (pad_top, pad_left, new_h, new_w),
    }

    return tensor, meta


def predict(
    model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Run model inference to produce a class mask.

    Args:
        model: SegFormer model for semantic segmentation.
        image_tensor: Preprocessed image tensor of shape (1, 3, H, W).
        device: Device to run inference on.

    Returns:
        2D uint8 numpy array of shape (INPUT_SIZE, INPUT_SIZE) where each
        pixel value is a class ID (0 = background).
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
        logits = outputs.logits  # (1, NUM_CLASSES, H/4, W/4)

        # SegFormer outputs at 1/4 resolution; interpolate to full size
        logits = F.interpolate(
            logits,
            size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        # Class prediction per pixel
        mask = logits.argmax(dim=1).squeeze(0)  # (INPUT_SIZE, INPUT_SIZE)

    return mask.cpu().numpy().astype(np.uint8)


def analyze_floorplan(
    model: torch.nn.Module, image_path: str, device: torch.device
) -> dict:
    """Full inference pipeline: image -> segmentation mask -> polygon JSON.

    Args:
        model: Loaded SegFormer model.
        image_path: Path to the input image.
        device: Device for inference.

    Returns:
        Dict with keys:
          version: 3
          image_width_meters: 0
          elements: dict mapping class names to polygon lists
    """
    tensor, meta = preprocess_image(image_path)
    mask = predict(model, tensor, device)

    polygons = mask_to_polygons(
        mask,
        original_size=meta["original_size"],
        padding=meta["padding"],
    )

    return {
        "version": 3,
        "image_width_meters": 0,
        "elements": polygons,
    }


def load_model(model_path: str, device: torch.device):
    """Load a saved SegFormer model from a directory.

    Args:
        model_path: Path to the saved model directory.
        device: Device to load the model onto.

    Returns:
        The loaded model in eval mode.
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_path,
        num_labels=NUM_CLASSES,
    )
    model.to(device)
    model.eval()
    return model


def save_mask_png(mask: np.ndarray, output_path: str) -> None:
    """Save a class mask as a visible grayscale PNG.

    Pixel values are scaled by 255 // NUM_CLASSES so that different
    classes are visually distinguishable.

    Args:
        mask: 2D uint8 array of class IDs.
        output_path: Path to write the PNG file.
    """
    scale = 255 // NUM_CLASSES
    visible = (mask * scale).astype(np.uint8)
    img = Image.fromarray(visible, mode="L")
    img.save(output_path)


def process_single(
    model: torch.nn.Module,
    image_path: str,
    output_path: str,
    device: torch.device,
    save_mask: bool = False,
) -> None:
    """Process a single image and write the result JSON.

    Args:
        model: Loaded model.
        image_path: Input image path.
        output_path: Output JSON path.
        device: Inference device.
        save_mask: If True, also save the mask as a PNG alongside the JSON.
    """
    print(f"Processing: {image_path}")

    tensor, meta = preprocess_image(image_path)
    mask = predict(model, tensor, device)

    polygons = mask_to_polygons(
        mask,
        original_size=meta["original_size"],
        padding=meta["padding"],
    )
    result = {
        "version": 3,
        "image_width_meters": 0,
        "elements": polygons,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {output_path}")

    if save_mask:
        mask_path = os.path.splitext(output_path)[0] + "_mask.png"
        save_mask_png(mask, mask_path)
        print(f"Saved mask: {mask_path}")


def process_batch(
    model: torch.nn.Module,
    image_dir: str,
    output_dir: str,
    device: torch.device,
    save_mask: bool = False,
) -> None:
    """Process all images in a directory.

    Args:
        model: Loaded model.
        image_dir: Directory containing input images.
        output_dir: Directory to write output JSON files.
        device: Inference device.
        save_mask: If True, also save masks as PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    image_files = sorted(
        f
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in extensions
    )

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images in {image_dir}")

    for fname in image_files:
        image_path = os.path.join(image_dir, fname)
        base_name = os.path.splitext(fname)[0]
        output_path = os.path.join(output_dir, base_name + ".json")

        print(f"Processing: {fname}")
        result = analyze_floorplan(model, image_path, device)

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> {output_path}")

        if save_mask:
            tensor, meta = preprocess_image(image_path)
            mask = predict(model, tensor, device)
            mask_path = os.path.join(output_dir, base_name + "_mask.png")
            save_mask_png(mask, mask_path)
            print(f"  -> {mask_path}")

    print(f"Done. Processed {len(image_files)} images.")


def main():
    parser = argparse.ArgumentParser(
        description="Floorplan segmentation inference: image -> polygon JSON"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Single image path (use with --output)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory of images for batch mode (use with --output-dir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (single image mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (batch mode)",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Also save the segmentation mask as a PNG",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.image and args.image_dir:
        parser.error("Cannot use both --image and --image-dir")
    if not args.image and not args.image_dir:
        parser.error("Must specify either --image or --image-dir")

    if args.image and not args.output:
        # Default output: same name with .json extension
        args.output = os.path.splitext(args.image)[0] + ".json"

    if args.image_dir and not args.output_dir:
        # Default output dir: image_dir + "_output"
        args.output_dir = args.image_dir.rstrip(os.sep) + "_output"

    # Device selection
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.model}")
    model = load_model(args.model, device)
    print("Model loaded.")

    # Run inference
    if args.image:
        process_single(model, args.image, args.output, device, args.save_mask)
    else:
        process_batch(
            model, args.image_dir, args.output_dir, device, args.save_mask
        )


if __name__ == "__main__":
    main()
