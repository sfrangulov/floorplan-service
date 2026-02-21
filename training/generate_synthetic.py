"""Render synthetic floorplan images and segmentation masks."""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from config import INPUT_SIZE
from floorplan_generator import Layout, generate_layout

# Class IDs matching config.py CLASSES
CLASS_BG = 0
CLASS_WALL = 1
CLASS_DOOR = 2
CLASS_WINDOW = 3
CLASS_BALCONY = 4
CLASS_BALCONY_WINDOW = 5
CLASS_BEDROOM = 6
CLASS_LIVING_ROOM = 7
CLASS_KITCHEN = 8
CLASS_BATHROOM = 9

ROOM_TYPE_TO_CLASS = {
    "bedroom": CLASS_BEDROOM,
    "living_room": CLASS_LIVING_ROOM,
    "kitchen": CLASS_KITCHEN,
    "bathroom": CLASS_BATHROOM,
}


def random_style() -> dict:
    """Generate a random visual style for rendering."""
    # Line/wall color: black, dark gray, or navy
    line_colors = [
        (0, 0, 0),          # black
        (40, 40, 40),       # dark gray
        (60, 60, 60),       # gray
        (30, 30, 60),       # navy
        (20, 20, 50),       # dark navy
    ]
    line_color = random.choice(line_colors)

    # Room fill colors (light pastel tones)
    room_fills = {
        "bedroom": (
            random.randint(220, 245),
            random.randint(220, 240),
            random.randint(230, 250),
        ),
        "living_room": (
            random.randint(230, 250),
            random.randint(235, 250),
            random.randint(220, 240),
        ),
        "kitchen": (
            random.randint(230, 250),
            random.randint(240, 255),
            random.randint(230, 245),
        ),
        "bathroom": (
            random.randint(220, 240),
            random.randint(230, 248),
            random.randint(240, 255),
        ),
        "balcony": (
            random.randint(235, 250),
            random.randint(235, 250),
            random.randint(230, 245),
        ),
    }

    return {
        "line_color": line_color,
        "wall_thickness": random.uniform(2, 5),
        "thickness_scale": random.uniform(0.8, 1.5),
        "room_fills": room_fills,
        "noise_level": random.uniform(0, 8),
        "jpeg_quality": random.randint(60, 95),
    }


def render_image(layout: Layout, style: dict) -> np.ndarray:
    """Render a floorplan layout as an RGB image.

    Args:
        layout: The floorplan layout to render.
        style: Visual style parameters from random_style().

    Returns:
        RGB image as uint8 numpy array of shape (H, W, 3).
    """
    h, w = layout.height, layout.width
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    line_color = style["line_color"]
    wall_t = max(1, int(style["wall_thickness"] * style["thickness_scale"]))
    fills = style["room_fills"]

    # --- Draw room fills ---
    for room in layout.rooms:
        color = fills.get(room.room_type, (240, 240, 240))
        pts = np.array(room.polygon, dtype=np.int32)
        cv2.fillPoly(img, [pts], color)

    # --- Draw balcony fill ---
    if layout.balcony is not None:
        color = fills.get("balcony", (240, 240, 235))
        pts = np.array(layout.balcony.polygon, dtype=np.int32)
        cv2.fillPoly(img, [pts], color)

    # --- Draw walls ---
    for wall in layout.walls:
        thickness = wall_t + (1 if wall.is_exterior else 0)
        cv2.line(
            img,
            (int(wall.x1), int(wall.y1)),
            (int(wall.x2), int(wall.y2)),
            line_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    # --- Draw balcony outline ---
    if layout.balcony is not None:
        b = layout.balcony
        pts = np.array(b.polygon, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=line_color,
                       thickness=max(1, wall_t - 1), lineType=cv2.LINE_AA)

    # --- Draw doors (clear wall segment + arc) ---
    for door in layout.doors:
        dx, dy = int(door.x), int(door.y)
        dw, dh = int(door.width), int(door.height)

        # Clear the wall under the door (white gap)
        cv2.rectangle(img, (dx, dy), (dx + dw, dy + dh), (255, 255, 255), -1)

        # Draw door arc
        if dh > dw:
            # Vertical wall door - arc swings horizontally
            arc_center = (dx + dw // 2, dy)
            arc_radius = dh
            cv2.ellipse(
                img, arc_center, (arc_radius, arc_radius),
                0, 0, 90, line_color, 1, cv2.LINE_AA,
            )
        else:
            # Horizontal wall door - arc swings vertically
            arc_center = (dx, dy + dh // 2)
            arc_radius = dw
            cv2.ellipse(
                img, arc_center, (arc_radius, arc_radius),
                0, -90, 0, line_color, 1, cv2.LINE_AA,
            )

    # --- Draw windows (parallel lines) ---
    for window in layout.windows:
        wx, wy = int(window.x), int(window.y)
        ww, wh = int(window.width), int(window.height)

        # Clear the wall segment under the window
        cv2.rectangle(img, (wx, wy), (wx + ww, wy + wh), (255, 255, 255), -1)

        if wh > ww:
            # Vertical window (on left/right wall)
            mid_x = wx + ww // 2
            cv2.line(img, (mid_x - 1, wy), (mid_x - 1, wy + wh),
                     line_color, 1, cv2.LINE_AA)
            cv2.line(img, (mid_x + 1, wy), (mid_x + 1, wy + wh),
                     line_color, 1, cv2.LINE_AA)
        else:
            # Horizontal window (on top/bottom wall)
            mid_y = wy + wh // 2
            cv2.line(img, (wx, mid_y - 1), (wx + ww, mid_y - 1),
                     line_color, 1, cv2.LINE_AA)
            cv2.line(img, (wx, mid_y + 1), (wx + ww, mid_y + 1),
                     line_color, 1, cv2.LINE_AA)

    # --- Draw balcony window (filled rect with gap pattern) ---
    if layout.balcony_window is not None:
        bw = layout.balcony_window
        bx, by = int(bw.x), int(bw.y)
        bww, bwh = int(bw.width), int(bw.height)

        # Clear wall under balcony window
        cv2.rectangle(img, (bx, by), (bx + bww, by + bwh),
                       (255, 255, 255), -1)

        # Draw as rectangle outline with center line
        cv2.rectangle(img, (bx, by), (bx + bww, by + bwh),
                       line_color, 1, cv2.LINE_AA)
        mid_y = by + bwh // 2
        cv2.line(img, (bx, mid_y), (bx + bww, mid_y),
                 line_color, 1, cv2.LINE_AA)

    # --- Apply noise ---
    noise_level = style["noise_level"]
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # --- Apply JPEG compression artifacts ---
    quality = style["jpeg_quality"]
    if quality < 95:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", img, encode_param)
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img


def render_mask(layout: Layout) -> np.ndarray:
    """Render a pixel-perfect segmentation mask.

    Drawing order ensures correct class overlap:
    1. Room fills (class IDs 6-9)
    2. Balcony (4)
    3. Balcony window (5)
    4. Walls (1) on top of rooms
    5. Doors (2) on top of walls
    6. Windows (3) on top of walls

    Args:
        layout: The floorplan layout to render.

    Returns:
        Single-channel uint8 mask of shape (H, W) with class IDs.
    """
    h, w = layout.height, layout.width
    mask = np.zeros((h, w), dtype=np.uint8)  # 0 = background

    # 1. Room fills
    for room in layout.rooms:
        class_id = ROOM_TYPE_TO_CLASS.get(room.room_type, CLASS_BG)
        pts = np.array(room.polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], int(class_id))

    # 2. Balcony
    if layout.balcony is not None:
        pts = np.array(layout.balcony.polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], CLASS_BALCONY)

    # 3. Balcony window
    if layout.balcony_window is not None:
        bw = layout.balcony_window
        bx, by = int(bw.x), int(bw.y)
        bww, bwh = int(bw.width), int(bw.height)
        cv2.rectangle(mask, (bx, by), (bx + bww, by + bwh),
                       CLASS_BALCONY_WINDOW, -1)

    # 4. Walls on top of rooms
    for wall in layout.walls:
        thickness = max(2, int(wall.thickness))
        if wall.is_exterior:
            thickness += 1
        cv2.line(
            mask,
            (int(wall.x1), int(wall.y1)),
            (int(wall.x2), int(wall.y2)),
            CLASS_WALL,
            thickness=thickness,
        )

    # Balcony outline as wall
    if layout.balcony is not None:
        b = layout.balcony
        pts = np.array(b.polygon, dtype=np.int32)
        cv2.polylines(mask, [pts], isClosed=True, color=CLASS_WALL,
                       thickness=max(2, int(layout.walls[0].thickness) - 1))

    # 5. Doors on top of walls
    for door in layout.doors:
        dx, dy = int(door.x), int(door.y)
        dw, dh = int(door.width), int(door.height)
        cv2.rectangle(mask, (dx, dy), (dx + dw, dy + dh), CLASS_DOOR, -1)

    # 6. Windows on top of walls
    for window in layout.windows:
        wx, wy = int(window.x), int(window.y)
        ww, wh = int(window.width), int(window.height)
        cv2.rectangle(mask, (wx, wy), (wx + ww, wy + wh), CLASS_WINDOW, -1)

    return mask


def main():
    """CLI entrypoint: generate synthetic floorplan dataset."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic floorplan training data."
    )
    parser.add_argument(
        "--count", type=int, default=2000,
        help="Number of samples to generate (default: 2000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/prepared/synthetic/train",
        help="Output directory (default: data/prepared/synthetic/train)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directories
    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} synthetic floorplans...")
    print(f"Output: {out_dir.resolve()}")

    for i in range(args.count):
        layout = generate_layout(canvas_size=INPUT_SIZE)
        style = random_style()

        image = render_image(layout, style)
        mask = render_mask(layout)

        # Save image as PNG (mask must be lossless)
        img_path = img_dir / f"{i:05d}.png"
        mask_path = mask_dir / f"{i:05d}.png"

        cv2.imwrite(str(img_path), image)
        cv2.imwrite(str(mask_path), mask)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.count}")

    print(f"Done. Generated {args.count} image/mask pairs.")


if __name__ == "__main__":
    main()
