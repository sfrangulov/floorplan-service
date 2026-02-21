#!/usr/bin/env python3
"""Prepare CubiCasa5K dataset for segmentation model training.

Parses SVG annotations from CubiCasa5K, maps categories to our 10-class
schema, renders pixel masks at 512x512, and saves image/mask pairs in
train/val splits.

Usage:
    python prepare_cubicasa.py \\
        --data-dir data/cubicasa5k \\
        --output-dir data/prepared/cubicasa \\
        --val-split 0.1
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from xml.dom import minidom

import cv2
import numpy as np
from svgpathtools import parse_path

# Add parent so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CUBICASA_MAPPING, INPUT_SIZE, NUM_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# CubiCasa5K quality subdirectories
SUBDIRS = ["colorful", "high_quality", "high_quality_architectural"]

# Rendering constants
WALL_THICKNESS = 3


# ---------------------------------------------------------------------------
# SVG parsing helpers
# ---------------------------------------------------------------------------

def _get_elements(svg_doc, tag_name):
    """Get elements by tag name, handling both namespaced and plain SVG."""
    # Try with svg: namespace prefix first
    elements = svg_doc.getElementsByTagName(f"svg:{tag_name}")
    if not elements:
        # Fall back to plain tag name (no namespace prefix)
        elements = svg_doc.getElementsByTagName(tag_name)
    return elements


def _parse_points(points_str):
    """Parse SVG polygon points string into numpy arrays of X, Y coords.

    Handles formats like:
        "x1,y1 x2,y2 x3,y3 "
        "x1 y1 x2 y2 x3 y3"
    """
    points_str = points_str.strip()
    if not points_str:
        return np.array([]), np.array([])

    tokens = points_str.split()
    # Remove empty tokens
    tokens = [t for t in tokens if t]

    X, Y = [], []

    if "," in tokens[0]:
        # Format: "x,y x,y x,y"
        for token in tokens:
            parts = token.split(",")
            if len(parts) >= 2:
                X.append(float(parts[0]))
                Y.append(float(parts[1]))
    else:
        # Format: "x y x y x y"
        for i in range(0, len(tokens) - 1, 2):
            X.append(float(tokens[i]))
            Y.append(float(tokens[i + 1]))

    return np.array(X), np.array(Y)


def _polygon_to_mask(X, Y, height, width):
    """Rasterize polygon coordinates into a boolean mask using OpenCV."""
    if len(X) < 3 or len(Y) < 3:
        return np.zeros((height, width), dtype=bool)

    pts = np.column_stack((X, Y)).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _polyline_to_mask(X, Y, height, width, thickness=WALL_THICKNESS):
    """Rasterize a polyline (wall outline) with given thickness."""
    if len(X) < 2 or len(Y) < 2:
        return np.zeros((height, width), dtype=bool)

    pts = np.column_stack((X, Y)).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    # Draw the polygon outline with thickness, then also fill it
    # Walls in CubiCasa are polygons, so we fill them
    cv2.fillPoly(mask, [pts], 1)
    # Also draw thick border to ensure thin walls are visible
    cv2.polylines(mask, [pts], isClosed=True, color=1, thickness=thickness)
    return mask.astype(bool)


def _path_to_polygon(d_attr):
    """Convert SVG path d attribute to bounding polygon X, Y arrays."""
    try:
        path = parse_path(d_attr)
        minx, maxx, miny, maxy = path.bbox()
        X = np.array([minx, maxx, maxx, minx])
        Y = np.array([miny, miny, maxy, maxy])
        return X, Y
    except (ValueError, ZeroDivisionError):
        return np.array([]), np.array([])


def _get_child_elements(parent, tag_name):
    """Get child elements by tag, handling both namespaced and plain SVG.

    Tries svg:-prefixed tag first, falls back to plain tag name.
    """
    elements = parent.getElementsByTagName(f"svg:{tag_name}")
    if not elements:
        elements = parent.getElementsByTagName(tag_name)
    return elements


def _extract_polygons_from_group(group_elem):
    """Extract all polygon/path coordinate arrays from an SVG group element.

    Returns list of (X, Y) tuples.
    """
    polygons = []

    # Get polygon elements (try namespaced then plain)
    for poly in _get_child_elements(group_elem, "polygon"):
        points_str = poly.getAttribute("points")
        if points_str:
            X, Y = _parse_points(points_str)
            if len(X) >= 3:
                polygons.append((X, Y))

    # Get path elements
    for path_elem in _get_child_elements(group_elem, "path"):
        d_attr = path_elem.getAttribute("d")
        if d_attr:
            X, Y = _path_to_polygon(d_attr)
            if len(X) >= 3:
                polygons.append((X, Y))

    return polygons


# ---------------------------------------------------------------------------
# Main SVG -> mask rendering
# ---------------------------------------------------------------------------

def parse_svg_to_mask(svg_path, height, width):
    """Parse a CubiCasa5K model.svg and render a class mask.

    Rendering order (later overwrites earlier):
        1. Rooms (fill) - bedroom, living_room, kitchen, bathroom, balcony
        2. Walls (polylines with thickness)
        3. Doors and Windows (fill on top)

    Args:
        svg_path: Path to model.svg
        height: Image height (original resolution)
        width: Image width (original resolution)

    Returns:
        mask: np.ndarray of shape (height, width), dtype=uint8,
              values are class indices 0..NUM_CLASSES-1
    """
    mask = np.zeros((height, width), dtype=np.uint8)  # 0 = background

    try:
        svg_doc = minidom.parse(str(svg_path))
    except Exception as e:
        logger.warning("Failed to parse SVG %s: %s", svg_path, e)
        return mask

    # Collect elements by category
    rooms = []    # (class_idx, polygons)
    walls = []    # (class_idx, polygons)
    openings = [] # (class_idx, polygons) - doors and windows

    for group in _get_elements(svg_doc, "g"):
        group_id = group.getAttribute("id")
        group_class = group.getAttribute("class")

        # --- Walls (identified by id="Wall") ---
        if group_id == "Wall":
            polys = _extract_polygons_from_group(group)
            if polys:
                walls.append((CUBICASA_MAPPING["Wall"], polys))
            continue

        # --- Windows (identified by id="Window") ---
        if group_id == "Window":
            polys = _extract_polygons_from_group(group)
            if polys:
                openings.append((CUBICASA_MAPPING["Window"], polys))
            continue

        # --- Doors (identified by id="Door") ---
        if group_id == "Door":
            polys = _extract_polygons_from_group(group)
            if polys:
                openings.append((CUBICASA_MAPPING["Door"], polys))
            continue

        # --- Rooms (identified by class="Space RoomType") ---
        if group_class and "Space " in group_class:
            parts = group_class.split()
            # The room type is the second token: "Space Bedroom" -> "Bedroom"
            if len(parts) >= 2:
                room_type = parts[1]
                if room_type in CUBICASA_MAPPING:
                    class_idx = CUBICASA_MAPPING[room_type]
                    polys = _extract_polygons_from_group(group)
                    if polys:
                        rooms.append((class_idx, polys))

        # --- Railing (treat as Wall) ---
        if group_id == "Railing":
            polys = _extract_polygons_from_group(group)
            if polys and "Wall" in CUBICASA_MAPPING:
                walls.append((CUBICASA_MAPPING["Wall"], polys))

    # Render in order: rooms -> walls -> openings
    # 1. Rooms (filled polygons)
    for class_idx, polys in rooms:
        for X, Y in polys:
            region = _polygon_to_mask(X, Y, height, width)
            mask[region] = class_idx

    # 2. Walls (polylines with thickness for visibility)
    for class_idx, polys in walls:
        for X, Y in polys:
            region = _polyline_to_mask(X, Y, height, width, WALL_THICKNESS)
            mask[region] = class_idx

    # 3. Doors and Windows (filled, on top of walls)
    for class_idx, polys in openings:
        for X, Y in polys:
            region = _polygon_to_mask(X, Y, height, width)
            mask[region] = class_idx

    return mask


# ---------------------------------------------------------------------------
# Image resizing with aspect ratio preservation
# ---------------------------------------------------------------------------

def resize_with_padding(image, target_size, interpolation=cv2.INTER_LINEAR,
                        pad_value=255):
    """Resize image preserving aspect ratio, pad to target_size x target_size.

    Args:
        image: Input image (H, W) or (H, W, C)
        target_size: Target square dimension
        interpolation: OpenCV interpolation flag
        pad_value: Value for padding pixels (255=white for images, 0 for masks)

    Returns:
        Resized and padded image of shape (target_size, target_size[, C])
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Create padded canvas
    if len(image.shape) == 3:
        canvas = np.full((target_size, target_size, image.shape[2]),
                         pad_value, dtype=image.dtype)
    else:
        canvas = np.full((target_size, target_size),
                         pad_value, dtype=image.dtype)

    # Center the resized image on the canvas
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


# ---------------------------------------------------------------------------
# Dataset discovery and processing
# ---------------------------------------------------------------------------

def discover_samples(data_dir):
    """Discover all valid CubiCasa5K samples.

    Looks for sample directories under colorful/, high_quality/,
    high_quality_architectural/ that contain both F1_scaled.png and model.svg.

    Returns:
        List of Path objects pointing to sample directories.
    """
    data_dir = Path(data_dir)
    samples = []

    for subdir_name in SUBDIRS:
        subdir = data_dir / subdir_name
        if not subdir.is_dir():
            logger.warning("Subdirectory not found: %s", subdir)
            continue

        for sample_dir in sorted(subdir.iterdir()):
            if not sample_dir.is_dir():
                continue

            image_path = sample_dir / "F1_scaled.png"
            svg_path = sample_dir / "model.svg"

            if image_path.exists() and svg_path.exists():
                samples.append(sample_dir)
            else:
                logger.debug("Skipping incomplete sample: %s", sample_dir)

    logger.info("Discovered %d valid samples across %d subdirectories",
                len(samples), len(SUBDIRS))
    return samples


def process_sample(sample_dir, output_dir, split):
    """Process a single CubiCasa5K sample: parse SVG, render mask, save.

    Args:
        sample_dir: Path to sample directory containing F1_scaled.png and model.svg
        output_dir: Base output directory
        split: "train" or "val"

    Returns:
        True if processed successfully, False otherwise.
    """
    image_path = sample_dir / "F1_scaled.png"
    svg_path = sample_dir / "model.svg"

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("Failed to read image: %s", image_path)
        return False

    h, w = image.shape[:2]

    # Parse SVG and render mask at original resolution
    mask = parse_svg_to_mask(svg_path, h, w)

    # Resize image: preserve aspect ratio, pad to INPUT_SIZE with white
    image_resized = resize_with_padding(
        image, INPUT_SIZE, interpolation=cv2.INTER_LINEAR, pad_value=255
    )

    # Resize mask: use INTER_NEAREST to avoid interpolation artifacts
    mask_resized = resize_with_padding(
        mask, INPUT_SIZE, interpolation=cv2.INTER_NEAREST, pad_value=0
    )

    # Create a unique sample name from the directory hierarchy
    # e.g., "high_quality/42" -> "high_quality_42"
    parent_name = sample_dir.parent.name
    sample_name = f"{parent_name}_{sample_dir.name}"

    # Save to split directory
    split_dir = Path(output_dir) / split
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    image_out = images_dir / f"{sample_name}.png"
    mask_out = masks_dir / f"{sample_name}.png"

    cv2.imwrite(str(image_out), image_resized)
    cv2.imwrite(str(mask_out), mask_resized)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CubiCasa5K dataset for segmentation training."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/cubicasa5k",
        help="Path to CubiCasa5K dataset root (default: data/cubicasa5k)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/prepared/cubicasa",
        help="Output directory for prepared data (default: data/prepared/cubicasa)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.is_dir():
        logger.error("Data directory not found: %s", data_dir)
        logger.error(
            "Please clone CubiCasa5K dataset into %s first.", data_dir
        )
        sys.exit(1)

    # Discover all samples
    samples = discover_samples(data_dir)
    if not samples:
        logger.error("No valid samples found in %s", data_dir)
        sys.exit(1)

    # Shuffle and split
    random.seed(42)
    random.shuffle(samples)

    val_count = int(len(samples) * args.val_split)
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    logger.info(
        "Split: %d train, %d val (%.0f%% val)",
        len(train_samples),
        len(val_samples),
        args.val_split * 100,
    )

    # Process all samples
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for i, sample_dir in enumerate(train_samples):
        if process_sample(sample_dir, output_dir, "train"):
            success_count += 1
        else:
            fail_count += 1
        if (i + 1) % 100 == 0:
            logger.info("Train progress: %d/%d", i + 1, len(train_samples))

    for i, sample_dir in enumerate(val_samples):
        if process_sample(sample_dir, output_dir, "val"):
            success_count += 1
        else:
            fail_count += 1
        if (i + 1) % 100 == 0:
            logger.info("Val progress: %d/%d", i + 1, len(val_samples))

    logger.info(
        "Done. Processed %d samples (%d failed). Output: %s",
        success_count,
        fail_count,
        output_dir,
    )
    logger.info("  Train: %s/train/", output_dir)
    logger.info("  Val:   %s/val/", output_dir)
    logger.info("  Classes: %d, Input size: %d", NUM_CLASSES, INPUT_SIZE)


if __name__ == "__main__":
    main()
