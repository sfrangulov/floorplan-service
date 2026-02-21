# Floorplan Segmentation Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Gemini LLM-based floorplan analysis with a fine-tuned SegFormer semantic segmentation model that outputs pixel-perfect masks, vectorized into polygons.

**Architecture:** SegFormer-B2 encoder + U-Net-style decoder, trained on CubiCasa5K + synthetic data, deployed on Replicate, called from Node.js Fastify service via `replicate` npm package. Post-processing converts 10-class pixel masks to contour polygons in normalized 0-1000 JSON.

**Tech Stack:** Python 3.11, PyTorch, HuggingFace Transformers (SegFormer), OpenCV, Sharp (Node.js), Replicate (Cog), Google Colab (training)

---

### Task 1: Set up Python ML workspace

**Files:**
- Create: `training/requirements.txt`
- Create: `training/config.py`
- Create: `training/.gitignore`

**Step 1: Create training directory and requirements**

```
training/requirements.txt:
```
```text
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.36.0
datasets>=2.16.0
opencv-python-headless>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
albumentations>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
huggingface-hub>=0.19.0
```

**Step 2: Create config with class definitions and hyperparameters**

```
training/config.py:
```
```python
"""Segmentation model configuration."""

# Class definitions - must match design doc
CLASSES = [
    "background",      # 0
    "wall",            # 1
    "door",            # 2
    "window",          # 3
    "balcony",         # 4
    "balcony_window",  # 5
    "bedroom",         # 6
    "living_room",     # 7
    "kitchen",         # 8
    "bathroom",        # 9
]

NUM_CLASSES = len(CLASSES)

# CubiCasa5K category mapping -> our 10 classes
# CubiCasa has 80+ categories. Map relevant ones, rest -> background
CUBICASA_MAPPING = {
    "Wall": 1,
    "Door": 2,
    "Window": 3,
    "Balcony": 4,
    "Bedroom": 6,
    "LivingRoom": 7,
    "Kitchen": 8,
    "Bathroom": 9,
    "Toilet": 9,
    "Bath": 9,
}

# Model
MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
INPUT_SIZE = 512

# Training hyperparameters
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Tversky loss parameters (emphasize recall for thin walls)
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7

# Output
COORD_RANGE = 1000  # normalize coordinates to 0-1000
```

**Step 3: Create .gitignore for training artifacts**

```
training/.gitignore:
```
```text
__pycache__/
*.pyc
*.pth
*.pt
checkpoints/
wandb/
data/cubicasa5k/
data/synthetic/
outputs/
*.egg-info/
.venv/
```

**Step 4: Commit**

```bash
git add training/requirements.txt training/config.py training/.gitignore
git commit -m "feat: add Python ML training workspace with config"
```

---

### Task 2: CubiCasa5K data preparation

**Files:**
- Create: `training/prepare_cubicasa.py`

**Step 1: Write the CubiCasa5K download and conversion script**

This script:
1. Downloads CubiCasa5K dataset (or expects it in `training/data/cubicasa5k/`)
2. Parses SVG annotations
3. Maps 80+ CubiCasa categories to our 10 classes
4. Renders pixel masks at 512x512
5. Saves as image/mask pairs in train/val split

```
training/prepare_cubicasa.py:
```
```python
"""
Download and prepare CubiCasa5K dataset for our 10-class segmentation.

Usage:
    python prepare_cubicasa.py --data-dir data/cubicasa5k --output-dir data/prepared

Expects CubiCasa5K to be cloned into data/cubicasa5k/:
    git clone https://github.com/CubiCasa/CubiCasa5k data/cubicasa5k
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import CUBICASA_MAPPING, NUM_CLASSES, INPUT_SIZE


def parse_svg_annotation(svg_path: str) -> dict[str, list]:
    """Parse CubiCasa SVG annotation into category -> polygon lists."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    categories = {}
    for g in root.findall(".//svg:g", ns):
        class_name = g.get("class", "")
        if not class_name:
            continue

        polygons = []
        for polygon in g.findall(".//svg:polygon", ns):
            points_str = polygon.get("points", "")
            if not points_str:
                continue
            points = []
            for pair in points_str.strip().split():
                parts = pair.split(",")
                if len(parts) == 2:
                    points.append([float(parts[0]), float(parts[1])])
            if len(points) >= 3:
                polygons.append(np.array(points, dtype=np.float32))

        for path_el in g.findall(".//svg:path", ns):
            d = path_el.get("d", "")
            # Simple M/L path parsing (CubiCasa uses simple paths)
            coords = re.findall(r"[-\d.]+", d)
            if len(coords) >= 6:
                points = []
                for i in range(0, len(coords) - 1, 2):
                    points.append([float(coords[i]), float(coords[i + 1])])
                if len(points) >= 3:
                    polygons.append(np.array(points, dtype=np.float32))

        if polygons:
            categories[class_name] = polygons

    return categories


def render_mask(categories: dict, image_size: tuple[int, int]) -> np.ndarray:
    """Render category polygons into a single-channel class mask."""
    h, w = image_size
    mask = np.zeros((h, w), dtype=np.uint8)

    # Render rooms first (background layer), then structural elements on top
    for cubicasa_name, our_class_id in CUBICASA_MAPPING.items():
        if our_class_id not in {6, 7, 8, 9}:  # skip non-room classes
            continue
        if cubicasa_name in categories:
            for poly in categories[cubicasa_name]:
                pts = poly.astype(np.int32)
                cv2.fillPoly(mask, [pts], our_class_id)

    # Then structural on top
    for cubicasa_name, our_class_id in CUBICASA_MAPPING.items():
        if our_class_id in {6, 7, 8, 9}:  # skip room classes
            continue
        if cubicasa_name in categories:
            for poly in categories[cubicasa_name]:
                pts = poly.astype(np.int32)
                if our_class_id == 1:  # walls are lines, use polylines
                    cv2.polylines(mask, [pts], True, our_class_id, thickness=3)
                else:
                    cv2.fillPoly(mask, [pts], our_class_id)

    return mask


def process_sample(
    image_path: str,
    svg_path: str,
    output_dir: str,
    sample_id: str,
):
    """Process one CubiCasa5K sample: image + SVG -> resized image + mask."""
    # Load image
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Parse SVG
    categories = parse_svg_annotation(svg_path)

    # Render mask at original size
    mask = render_mask(categories, (orig_h, orig_w))

    # Resize both to INPUT_SIZE x INPUT_SIZE (preserve aspect ratio + pad)
    scale = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Pad to INPUT_SIZE x INPUT_SIZE
    img_padded = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (255, 255, 255))
    img_padded.paste(img_resized, (0, 0))

    mask_padded = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    mask_padded[:new_h, :new_w] = mask_resized

    # Save
    img_padded.save(os.path.join(output_dir, "images", f"{sample_id}.png"))
    cv2.imwrite(os.path.join(output_dir, "masks", f"{sample_id}.png"), mask_padded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/cubicasa5k")
    parser.add_argument("--output-dir", default="data/prepared/cubicasa")
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    # Find all samples (CubiCasa5K structure: colorful/high_quality/high_quality_architectural)
    samples = []
    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for sample_dir in sorted(category_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            image_path = sample_dir / "F1_scaled.png"
            svg_path = sample_dir / "model.svg"
            if image_path.exists() and svg_path.exists():
                samples.append((str(image_path), str(svg_path), sample_dir.name))

    print(f"Found {len(samples)} samples")

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    val_count = int(len(samples) * args.val_split)
    val_indices = set(indices[:val_count])

    for i, (img_path, svg_path, sample_id) in enumerate(tqdm(samples)):
        split = "val" if i in val_indices else "train"
        try:
            process_sample(img_path, svg_path, str(output_dir / split), sample_id)
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")


if __name__ == "__main__":
    main()
```

**Step 2: Test the script works with a dry run**

Run: `cd training && python prepare_cubicasa.py --help`
Expected: shows help text without errors

**Step 3: Commit**

```bash
git add training/prepare_cubicasa.py
git commit -m "feat: add CubiCasa5K data preparation script"
```

---

### Task 3: Synthetic floorplan generator

**Files:**
- Create: `training/generate_synthetic.py`
- Create: `training/floorplan_generator.py`

**Step 1: Write the floorplan layout generator**

This generates random apartment layouts with rooms, walls, doors, windows.

```
training/floorplan_generator.py:
```
```python
"""
Procedural floorplan layout generator.

Generates random apartment layouts as a set of rooms with walls, doors, and windows.
Each layout is represented as:
- rooms: list of {type, polygon} where polygon is a list of (x,y) points
- walls: list of {start, end, thickness} line segments
- doors: list of {position, width} on wall segments
- windows: list of {position, width} on exterior wall segments
- balcony: optional {polygon}
"""

import random
from dataclasses import dataclass, field


@dataclass
class Room:
    room_type: str  # bedroom, living_room, kitchen, bathroom
    x: float
    y: float
    w: float
    h: float

    @property
    def polygon(self):
        return [
            (self.x, self.y),
            (self.x + self.w, self.y),
            (self.x + self.w, self.y + self.h),
            (self.x, self.y + self.h),
        ]

    @property
    def center(self):
        return (self.x + self.w / 2, self.y + self.h / 2)


@dataclass
class Wall:
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float = 3.0
    is_exterior: bool = False


@dataclass
class Door:
    x: float
    y: float
    width: float
    height: float


@dataclass
class Window:
    x: float
    y: float
    width: float
    height: float


@dataclass
class Layout:
    width: float  # total image width
    height: float  # total image height
    rooms: list[Room] = field(default_factory=list)
    walls: list[Wall] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)
    balcony: Room | None = None
    balcony_window: tuple | None = None  # (x1, y1, x2, y2, thickness)


def generate_layout(
    canvas_size: int = 512,
    num_bedrooms: int | None = None,
    has_balcony: bool | None = None,
) -> Layout:
    """Generate a random apartment layout."""

    if num_bedrooms is None:
        num_bedrooms = random.choice([1, 1, 2, 2, 2, 3])
    if has_balcony is None:
        has_balcony = random.random() < 0.7

    margin = int(canvas_size * 0.08)
    wall_t = random.uniform(2.5, 5.0)

    # Apartment bounding box
    apt_x = margin
    apt_y = margin + (int(canvas_size * 0.12) if has_balcony else 0)
    apt_w = canvas_size - 2 * margin
    apt_h = canvas_size - margin - apt_y

    layout = Layout(width=canvas_size, height=canvas_size)

    # Generate room grid
    rooms = _generate_rooms(apt_x, apt_y, apt_w, apt_h, num_bedrooms, wall_t)
    layout.rooms = rooms

    # Generate walls from room boundaries
    layout.walls = _generate_walls(rooms, apt_x, apt_y, apt_w, apt_h, wall_t)

    # Generate doors between adjacent rooms
    layout.doors = _generate_doors(rooms, wall_t)

    # Generate windows on exterior walls
    layout.windows = _generate_windows(rooms, apt_x, apt_y, apt_w, apt_h, wall_t)

    # Balcony
    if has_balcony:
        bal_h = int(canvas_size * random.uniform(0.08, 0.14))
        balcony = Room("balcony", apt_x, apt_y - bal_h - wall_t, apt_w, bal_h)
        layout.balcony = balcony

        # Balcony window (glass partition)
        bw_t = wall_t * 1.5
        layout.balcony_window = (apt_x, apt_y - bw_t, apt_x + apt_w, apt_y, bw_t)

    return layout


def _generate_rooms(
    apt_x, apt_y, apt_w, apt_h, num_bedrooms, wall_t
) -> list[Room]:
    """Generate rooms using a simple grid-based approach."""
    rooms = []
    t = wall_t

    # Split apartment into top and bottom halves
    split_y = apt_y + apt_h * random.uniform(0.45, 0.55)

    # Top row: bedrooms + living room
    top_h = split_y - apt_y - t
    bedroom_total_w = apt_w * random.uniform(0.35, 0.5)
    bedroom_w = bedroom_total_w / num_bedrooms

    for i in range(num_bedrooms):
        rooms.append(Room(
            "bedroom",
            apt_x + i * (bedroom_w + t) + t,
            apt_y + t,
            bedroom_w - t,
            top_h,
        ))

    # Living room fills rest of top row
    living_x = apt_x + num_bedrooms * (bedroom_w + t) + t
    living_w = apt_w - (living_x - apt_x) - t
    rooms.append(Room("living_room", living_x, apt_y + t, living_w, top_h))

    # Bottom row: kitchen + bathroom(s)
    bottom_h = apt_y + apt_h - split_y - t

    # Kitchen (larger)
    kitchen_w = apt_w * random.uniform(0.35, 0.5)
    rooms.append(Room("kitchen", apt_x + t, split_y + t, kitchen_w - t, bottom_h - t))

    # Bathroom
    bath_w = apt_w * random.uniform(0.15, 0.25)
    bath_x = apt_x + kitchen_w + t
    rooms.append(Room("bathroom", bath_x, split_y + t, bath_w - t, bottom_h - t))

    return rooms


def _generate_walls(rooms, apt_x, apt_y, apt_w, apt_h, wall_t) -> list[Wall]:
    """Generate wall segments from room boundaries."""
    walls = []

    # Exterior walls
    walls.append(Wall(apt_x, apt_y, apt_x + apt_w, apt_y, wall_t, True))  # top
    walls.append(Wall(apt_x, apt_y + apt_h, apt_x + apt_w, apt_y + apt_h, wall_t, True))  # bottom
    walls.append(Wall(apt_x, apt_y, apt_x, apt_y + apt_h, wall_t, True))  # left
    walls.append(Wall(apt_x + apt_w, apt_y, apt_x + apt_w, apt_y + apt_h, wall_t, True))  # right

    # Interior walls between rooms
    for i, r1 in enumerate(rooms):
        for r2 in rooms[i + 1:]:
            # Check if rooms share a vertical wall
            if abs(r1.x + r1.w - r2.x) < wall_t * 2:
                x = (r1.x + r1.w + r2.x) / 2
                y_start = max(r1.y, r2.y)
                y_end = min(r1.y + r1.h, r2.y + r2.h)
                if y_end > y_start:
                    walls.append(Wall(x, y_start, x, y_end, wall_t))

            # Check if rooms share a horizontal wall
            if abs(r1.y + r1.h - r2.y) < wall_t * 2:
                y = (r1.y + r1.h + r2.y) / 2
                x_start = max(r1.x, r2.x)
                x_end = min(r1.x + r1.w, r2.x + r2.w)
                if x_end > x_start:
                    walls.append(Wall(x_start, y, x_end, y, wall_t))

    return walls


def _generate_doors(rooms, wall_t) -> list[Door]:
    """Place doors between adjacent rooms."""
    doors = []
    door_w = random.uniform(12, 20)
    door_h = wall_t * 1.2

    for i, r1 in enumerate(rooms):
        for r2 in rooms[i + 1:]:
            # Vertical adjacency
            if abs(r1.x + r1.w - r2.x) < wall_t * 3:
                y_start = max(r1.y, r2.y)
                y_end = min(r1.y + r1.h, r2.y + r2.h)
                if y_end - y_start > door_w * 2:
                    dy = random.uniform(y_start + door_w, y_end - door_w)
                    dx = (r1.x + r1.w + r2.x) / 2 - door_h / 2
                    doors.append(Door(dx, dy - door_w / 2, door_h, door_w))

            # Horizontal adjacency
            if abs(r1.y + r1.h - r2.y) < wall_t * 3:
                x_start = max(r1.x, r2.x)
                x_end = min(r1.x + r1.w, r2.x + r2.w)
                if x_end - x_start > door_w * 2:
                    dx = random.uniform(x_start + door_w, x_end - door_w)
                    dy = (r1.y + r1.h + r2.y) / 2 - door_h / 2
                    doors.append(Door(dx - door_w / 2, dy, door_w, door_h))

    return doors


def _generate_windows(rooms, apt_x, apt_y, apt_w, apt_h, wall_t) -> list[Window]:
    """Place windows on exterior walls for rooms that touch the exterior."""
    windows = []
    window_h = wall_t * 1.3

    for room in rooms:
        if room.room_type in ("bathroom",):
            continue  # bathrooms usually don't have windows

        # Top exterior wall
        if abs(room.y - apt_y - wall_t) < wall_t * 2:
            win_w = room.w * random.uniform(0.3, 0.6)
            win_x = room.x + (room.w - win_w) / 2
            windows.append(Window(win_x, apt_y - window_h / 2 + wall_t / 2, win_w, window_h))

        # Bottom exterior wall
        if abs(room.y + room.h - (apt_y + apt_h - wall_t)) < wall_t * 2:
            win_w = room.w * random.uniform(0.3, 0.6)
            win_x = room.x + (room.w - win_w) / 2
            windows.append(Window(win_x, apt_y + apt_h - window_h / 2 - wall_t / 2, win_w, window_h))

    return windows
```

**Step 2: Write the renderer + mask generator**

```
training/generate_synthetic.py:
```
```python
"""
Generate synthetic floorplan images with pixel-perfect segmentation masks.

Usage:
    python generate_synthetic.py --count 2000 --output-dir data/prepared/synthetic/train
    python generate_synthetic.py --count 500 --output-dir data/prepared/synthetic/val --seed 999
"""

import argparse
import os
import random

import cv2
import numpy as np
from tqdm import tqdm

from config import INPUT_SIZE
from floorplan_generator import generate_layout, Layout


def render_image(layout: Layout, style: dict) -> np.ndarray:
    """Render a floorplan layout as a realistic-looking image."""
    size = INPUT_SIZE
    img = np.ones((size, size, 3), dtype=np.uint8) * 255  # white background

    line_color = style["line_color"]
    wall_t = max(2, int(style["wall_thickness"]))

    # Draw room fills (light gray for rooms)
    for room in layout.rooms:
        pts = np.array(room.polygon, dtype=np.int32)
        fill = style.get("room_fill", (245, 245, 245))
        cv2.fillPoly(img, [pts], fill)

    # Draw walls
    for wall in layout.walls:
        pt1 = (int(wall.x1), int(wall.y1))
        pt2 = (int(wall.x2), int(wall.y2))
        t = max(2, int(wall.thickness * style["thickness_scale"]))
        cv2.line(img, pt1, pt2, line_color, t)

    # Draw doors (gap in wall + arc)
    for door in layout.doors:
        x, y, w, h = int(door.x), int(door.y), int(door.width), int(door.height)
        # Clear wall area (white rectangle)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        # Draw door arc
        if w > h:  # horizontal door
            cv2.ellipse(img, (x, y + h // 2), (w, w), 0, -90, 0, line_color, 1)
        else:  # vertical door
            cv2.ellipse(img, (x + w // 2, y), (h, h), 0, 0, 90, line_color, 1)

    # Draw windows (parallel lines)
    for window in layout.windows:
        x, y, w, h = int(window.x), int(window.y), int(window.width), int(window.height)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        cv2.line(img, (x, y + h // 3), (x + w, y + h // 3), line_color, 1)
        cv2.line(img, (x, y + 2 * h // 3), (x + w, y + 2 * h // 3), line_color, 1)

    # Balcony
    if layout.balcony:
        pts = np.array(layout.balcony.polygon, dtype=np.int32)
        cv2.polylines(img, [pts], True, line_color, max(1, wall_t // 2))

    # Balcony window
    if layout.balcony_window:
        x1, y1, x2, y2, t = layout.balcony_window
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), line_color, max(1, int(t * style["thickness_scale"])))

    # Add noise if configured
    if style.get("noise_level", 0) > 0:
        noise = np.random.normal(0, style["noise_level"], img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if style.get("jpeg_quality"):
        # Simulate JPEG compression artifacts
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, style["jpeg_quality"]])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return img


def render_mask(layout: Layout) -> np.ndarray:
    """Render pixel-perfect segmentation mask from layout."""
    size = INPUT_SIZE
    mask = np.zeros((size, size), dtype=np.uint8)  # 0 = background

    # 1. Room fills (IDs 6-9) - draw first so walls overlay
    for room in layout.rooms:
        pts = np.array(room.polygon, dtype=np.int32)
        class_id = {
            "bedroom": 6,
            "living_room": 7,
            "kitchen": 8,
            "bathroom": 9,
        }.get(room.room_type, 0)
        if class_id > 0:
            cv2.fillPoly(mask, [pts], class_id)

    # 2. Balcony (ID 4)
    if layout.balcony:
        pts = np.array(layout.balcony.polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 4)

    # 3. Balcony window (ID 5)
    if layout.balcony_window:
        x1, y1, x2, y2, t = layout.balcony_window
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 5, -1)

    # 4. Walls (ID 1) - on top of rooms
    for wall in layout.walls:
        pt1 = (int(wall.x1), int(wall.y1))
        pt2 = (int(wall.x2), int(wall.y2))
        t = max(2, int(wall.thickness))
        cv2.line(mask, pt1, pt2, 1, t)

    # 5. Doors (ID 2) - on top of walls
    for door in layout.doors:
        x, y, w, h = int(door.x), int(door.y), int(door.width), int(door.height)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 2, -1)

    # 6. Windows (ID 3) - on top of walls
    for window in layout.windows:
        x, y, w, h = int(window.x), int(window.y), int(window.width), int(window.height)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 3, -1)

    return mask


def random_style() -> dict:
    """Generate random visual style for a floorplan."""
    return {
        "line_color": random.choice([
            (0, 0, 0),           # black
            (40, 40, 40),        # dark gray
            (60, 60, 80),        # dark blue-gray
            (30, 30, 60),        # navy
        ]),
        "wall_thickness": random.uniform(2, 5),
        "thickness_scale": random.uniform(0.8, 1.5),
        "room_fill": random.choice([
            (255, 255, 255),     # white
            (245, 245, 245),     # light gray
            (250, 248, 240),     # warm white
            (240, 245, 250),     # cool white
        ]),
        "noise_level": random.choice([0, 0, 0, 2, 4, 6]),
        "jpeg_quality": random.choice([None, None, 85, 75, 60]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--output-dir", default="data/prepared/synthetic/train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    images_dir = os.path.join(args.output_dir, "images")
    masks_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    for i in tqdm(range(args.count), desc="Generating"):
        layout = generate_layout(canvas_size=INPUT_SIZE)
        style = random_style()

        img = render_image(layout, style)
        mask = render_mask(layout)

        cv2.imwrite(os.path.join(images_dir, f"synth_{i:05d}.png"), img)
        cv2.imwrite(os.path.join(masks_dir, f"synth_{i:05d}.png"), mask)

    print(f"Generated {args.count} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
```

**Step 3: Test the generator locally**

Run: `cd training && python generate_synthetic.py --count 5 --output-dir data/test_gen`
Expected: creates 5 image/mask pairs in `data/test_gen/images/` and `data/test_gen/masks/`

Visually inspect: open a mask image - values should be 0-9 (will appear nearly black; multiply by 25 to visualize)

**Step 4: Commit**

```bash
git add training/floorplan_generator.py training/generate_synthetic.py
git commit -m "feat: add synthetic floorplan generator with mask rendering"
```

---

### Task 4: SegFormer training script

**Files:**
- Create: `training/train.py`
- Create: `training/dataset.py`
- Create: `training/loss.py`

**Step 1: Write the dataset loader**

```
training/dataset.py:
```
```python
"""PyTorch dataset for floorplan segmentation."""

import os

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import INPUT_SIZE, NUM_CLASSES


class FloorplanDataset(Dataset):
    """Loads image/mask pairs for segmentation training."""

    def __init__(self, data_dirs: list[str], transform=None):
        """
        Args:
            data_dirs: list of directories, each containing images/ and masks/ subdirs
            transform: albumentations transform
        """
        self.samples = []
        for data_dir in data_dirs:
            images_dir = os.path.join(data_dir, "images")
            masks_dir = os.path.join(data_dir, "masks")
            if not os.path.isdir(images_dir):
                continue
            for fname in sorted(os.listdir(images_dir)):
                if not fname.endswith(".png"):
                    continue
                img_path = os.path.join(images_dir, fname)
                mask_path = os.path.join(masks_dir, fname)
                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure correct size
        if image.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
            mask = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)

        # Clamp mask values to valid class range
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return image, mask


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.GaussNoise(var_limit=(5, 25), p=0.3),
        A.ElasticTransform(alpha=20, sigma=5, p=0.2),
    ])


def get_val_transform():
    return None  # no augmentation for validation
```

**Step 2: Write the Tversky loss**

```
training/loss.py:
```
```python
"""Tversky loss for segmentation with class imbalance."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, TVERSKY_ALPHA, TVERSKY_BETA


class TverskyLoss(nn.Module):
    """
    Tversky loss: generalizes Dice loss with asymmetric FP/FN weighting.
    alpha < beta emphasizes recall (fewer false negatives) - good for thin walls.
    """

    def __init__(self, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) class indices
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_one_hot = F.one_hot(targets, NUM_CLASSES)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Per-class Tversky index
        dims = (0, 2, 3)  # reduce over batch, height, width
        tp = (probs * targets_one_hot).sum(dim=dims)
        fp = (probs * (1 - targets_one_hot)).sum(dim=dims)
        fn = ((1 - probs) * targets_one_hot).sum(dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Average over classes (skip background class 0)
        return 1 - tversky[1:].mean()


class CombinedLoss(nn.Module):
    """Cross-entropy + Tversky loss."""

    def __init__(self, ce_weight=0.5, tversky_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.tversky = TverskyLoss()
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + self.tversky_weight * self.tversky(logits, targets)
```

**Step 3: Write the training script**

```
training/train.py:
```
```python
"""
Train SegFormer for floorplan segmentation.

Usage:
    python train.py --data-dirs data/prepared/cubicasa/train data/prepared/synthetic/train \
                    --val-dirs data/prepared/cubicasa/val data/prepared/synthetic/val \
                    --output-dir checkpoints/segformer-floorplan \
                    --epochs 100
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

from config import (
    MODEL_NAME, NUM_CLASSES, LEARNING_RATE,
    WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
)
from dataset import FloorplanDataset, get_train_transform, get_val_transform
from loss import CombinedLoss


def compute_miou(preds, targets, num_classes):
    """Compute mean IoU across classes (ignoring background)."""
    ious = []
    for c in range(1, num_classes):  # skip background
        pred_c = (preds == c)
        target_c = (targets == c)
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(pixel_values=images)
        logits = outputs.logits  # (B, C, H/4, W/4) - SegFormer outputs at 1/4 res

        # Upsample logits to match mask size
        logits = torch.nn.functional.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = criterion(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(pixel_values=images)
        logits = outputs.logits

        logits = torch.nn.functional.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    miou = compute_miou(all_preds, all_targets, NUM_CLASSES)

    return total_loss / len(loader), miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", nargs="+", required=True)
    parser.add_argument("--val-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="checkpoints/segformer-floorplan")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Data
    train_dataset = FloorplanDataset(args.data_dirs, transform=get_train_transform())
    val_dataset = FloorplanDataset(args.val_dirs, transform=get_val_transform())
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # Training setup
    criterion = CombinedLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = PolynomialLR(optimizer, total_iters=args.epochs, power=1.0)

    best_miou = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_miou = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_miou:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            model.save_pretrained(os.path.join(args.output_dir, "best"))
            print(f"  -> New best mIoU: {best_miou:.4f}")
        else:
            patience_counter += 1

        # Save latest
        model.save_pretrained(os.path.join(args.output_dir, "latest"))

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} "
                  f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break

    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best')}")


if __name__ == "__main__":
    main()
```

**Step 4: Commit**

```bash
git add training/dataset.py training/loss.py training/train.py
git commit -m "feat: add SegFormer training pipeline with Tversky loss"
```

---

### Task 5: Post-processing - mask to polygon vectorization

**Files:**
- Create: `training/vectorize.py`
- Test: `training/test_vectorize.py`

**Step 1: Write the failing test**

```
training/test_vectorize.py:
```
```python
"""Tests for mask -> polygon vectorization."""

import numpy as np
import pytest
from vectorize import mask_to_polygons


def test_single_wall_rectangle():
    """A horizontal wall rectangle should produce one wall polygon."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:45, 10:90] = 1  # wall class

    result = mask_to_polygons(mask, original_size=(100, 100))
    assert "wall" in result
    assert len(result["wall"]) == 1
    # Polygon should be in 0-1000 coords
    for point in result["wall"][0]:
        assert 0 <= point[0] <= 1000
        assert 0 <= point[1] <= 1000


def test_room_fills():
    """Room areas should produce room polygons."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:50, 10:50] = 6  # bedroom
    mask[10:50, 55:90] = 7  # living_room

    result = mask_to_polygons(mask, original_size=(100, 100))
    assert len(result["bedroom"]) == 1
    assert len(result["living_room"]) == 1


def test_empty_mask():
    """Empty mask should return empty arrays for all classes."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = mask_to_polygons(mask, original_size=(100, 100))
    for key in ("wall", "door", "window", "bedroom", "living_room", "kitchen", "bathroom"):
        assert result[key] == []


def test_coordinates_normalized_to_1000():
    """All output coordinates should be in 0-1000 range."""
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:110, 50:400] = 1  # wall

    result = mask_to_polygons(mask, original_size=(512, 512))
    for poly in result["wall"]:
        for x, y in poly:
            assert 0 <= x <= 1000, f"x={x} out of range"
            assert 0 <= y <= 1000, f"y={y} out of range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 2: Run tests to verify they fail**

Run: `cd training && python -m pytest test_vectorize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vectorize'`

**Step 3: Write the vectorization module**

```
training/vectorize.py:
```
```python
"""
Convert segmentation mask to polygon JSON.

Takes a (H, W) numpy array with class IDs 0-9 and converts to
a dict of class -> list of polygons in normalized 0-1000 coordinates.
"""

import cv2
import numpy as np

from config import COORD_RANGE

# Class ID -> output key mapping
CLASS_KEY = {
    1: "wall",
    2: "door",
    3: "window",
    4: "balcony",
    5: "balcony_window",
    6: "bedroom",
    7: "living_room",
    8: "kitchen",
    9: "bathroom",
}

# Minimum contour area (in pixels) to keep
MIN_AREA = {
    "wall": 20,
    "door": 15,
    "window": 15,
    "balcony": 50,
    "balcony_window": 20,
    "bedroom": 100,
    "living_room": 100,
    "kitchen": 100,
    "bathroom": 50,
}

# Douglas-Peucker epsilon per class (in pixels)
SIMPLIFY_EPSILON = {
    "wall": 1.5,
    "door": 2.0,
    "window": 2.0,
    "balcony": 2.0,
    "balcony_window": 2.0,
    "bedroom": 3.0,
    "living_room": 3.0,
    "kitchen": 3.0,
    "bathroom": 3.0,
}


def mask_to_polygons(
    mask: np.ndarray,
    original_size: tuple[int, int] | None = None,
    padding: tuple[int, int, int, int] | None = None,
) -> dict:
    """
    Convert a segmentation mask to polygons.

    Args:
        mask: (H, W) uint8 array with class IDs 0-9
        original_size: (orig_h, orig_w) of the input image before resize+pad.
                       If None, uses mask dimensions.
        padding: (pad_top, pad_left, content_h, content_w) - the resized content
                 area within the padded mask. If None, assumes no padding.

    Returns:
        dict with keys: wall, door, window, balcony, balcony_window,
                        bedroom, living_room, kitchen, bathroom
        Each value is a list of polygons, where each polygon is a list of
        [x, y] pairs in 0-1000 normalized coordinates.
    """
    h, w = mask.shape[:2]

    if original_size is None:
        orig_h, orig_w = h, w
    else:
        orig_h, orig_w = original_size

    # Determine the content area within the padded mask
    if padding:
        pad_top, pad_left, content_h, content_w = padding
    else:
        pad_top, pad_left = 0, 0
        content_h, content_w = h, w

    result = {}
    for class_id, key in CLASS_KEY.items():
        # Extract binary mask for this class
        binary = (mask == class_id).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        min_area = MIN_AREA.get(key, 20)
        epsilon = SIMPLIFY_EPSILON.get(key, 2.0)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Simplify with Douglas-Peucker
            simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
            if len(simplified) < 3:
                continue

            # Convert to list of [x, y] points
            points = simplified.squeeze().tolist()
            if len(points) < 3:
                continue

            # Ensure it's a list of [x, y] pairs
            if isinstance(points[0], (int, float)):
                continue  # degenerate case

            # Transform coordinates:
            # 1. Remove padding offset
            # 2. Scale to original image space
            # 3. Normalize to 0-1000
            normalized_points = []
            for px, py in points:
                # Remove padding
                x = px - pad_left
                y = py - pad_top
                # Scale from content space to original
                x = x / content_w * orig_w
                y = y / content_h * orig_h
                # Normalize to 0-1000
                x = round(x / orig_w * COORD_RANGE, 1)
                y = round(y / orig_h * COORD_RANGE, 1)
                # Clamp
                x = max(0, min(COORD_RANGE, x))
                y = max(0, min(COORD_RANGE, y))
                normalized_points.append([x, y])

            # Close polygon if not closed
            if normalized_points[0] != normalized_points[-1]:
                normalized_points.append(normalized_points[0])

            polygons.append(normalized_points)

        result[key] = polygons

    return result
```

**Step 4: Run tests to verify they pass**

Run: `cd training && python -m pytest test_vectorize.py -v`
Expected: all 4 tests PASS

**Step 5: Commit**

```bash
git add training/vectorize.py training/test_vectorize.py
git commit -m "feat: add mask-to-polygon vectorization with tests"
```

---

### Task 6: Inference script (full pipeline: image -> JSON)

**Files:**
- Create: `training/predict.py`

**Step 1: Write the inference pipeline**

```
training/predict.py:
```
```python
"""
Full inference pipeline: image -> segmentation mask -> polygon JSON.

Usage:
    python predict.py --model checkpoints/segformer-floorplan/best \
                      --image ../test-data/U3laAgYJ.jpg \
                      --output result.json

    # Batch mode:
    python predict.py --model checkpoints/segformer-floorplan/best \
                      --image-dir ../test-data/ \
                      --output-dir results/
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from config import INPUT_SIZE, NUM_CLASSES
from vectorize import mask_to_polygons


def preprocess_image(image_path: str) -> tuple[torch.Tensor, dict]:
    """
    Load and preprocess image for SegFormer.

    Returns:
        tensor: (1, 3, H, W) normalized image tensor
        meta: dict with original_size and padding info for post-processing
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Resize preserving aspect ratio
    scale = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Pad to INPUT_SIZE x INPUT_SIZE
    img_padded = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (255, 255, 255))
    img_padded.paste(img_resized, (0, 0))

    # Convert to tensor
    arr = np.array(img_padded).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    meta = {
        "original_size": (orig_h, orig_w),
        "padding": (0, 0, new_h, new_w),
    }
    return tensor, meta


@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run segmentation inference. Returns (H, W) mask with class IDs."""
    image_tensor = image_tensor.to(device)
    outputs = model(pixel_values=image_tensor)
    logits = outputs.logits  # (1, C, H/4, W/4)

    # Upsample to full resolution
    logits = torch.nn.functional.interpolate(
        logits, size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear", align_corners=False,
    )

    mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return mask


def analyze_floorplan(model, image_path: str, device: torch.device) -> dict:
    """Full pipeline: image path -> JSON result."""
    image_tensor, meta = preprocess_image(image_path)
    mask = predict(model, image_tensor, device)

    result = mask_to_polygons(
        mask,
        original_size=meta["original_size"],
        padding=meta["padding"],
    )

    result["version"] = 3
    result["image_width_meters"] = 0  # TODO: estimate from image or pass as param

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved model directory")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--image-dir", help="Directory of images for batch prediction")
    parser.add_argument("--output", help="Output JSON path (single image mode)")
    parser.add_argument("--output-dir", help="Output directory (batch mode)")
    parser.add_argument("--save-mask", action="store_true",
                        help="Also save the raw mask as PNG")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    print(f"Loading model from {args.model}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, num_labels=NUM_CLASSES
    )
    model.to(device)

    if args.image:
        # Single image mode
        start = time.time()
        result = analyze_floorplan(model, args.image, device)
        elapsed = time.time() - start
        print(f"Inference time: {elapsed:.3f}s")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))

        if args.save_mask:
            image_tensor, meta = preprocess_image(args.image)
            mask = predict(model, image_tensor, device)
            mask_vis = (mask * (255 // NUM_CLASSES)).astype(np.uint8)
            mask_path = (args.output or "mask") + ".mask.png"
            cv2.imwrite(mask_path, mask_vis)
            print(f"Mask saved to {mask_path}")

    elif args.image_dir:
        # Batch mode
        out_dir = args.output_dir or "results"
        os.makedirs(out_dir, exist_ok=True)

        for fname in sorted(os.listdir(args.image_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            img_path = os.path.join(args.image_dir, fname)
            start = time.time()
            result = analyze_floorplan(model, img_path, device)
            elapsed = time.time() - start

            out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"{fname}: {elapsed:.3f}s -> {out_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add training/predict.py
git commit -m "feat: add full inference pipeline (image -> mask -> JSON)"
```

---

### Task 7: Replicate deployment (Cog container)

**Files:**
- Create: `training/cog.yaml`
- Create: `training/cog_predict.py`

**Step 1: Write the Cog configuration**

```
training/cog.yaml:
```
```yaml
build:
  python_version: "3.11"
  python_packages:
    - "torch>=2.1.0"
    - "torchvision>=0.16.0"
    - "transformers>=4.36.0"
    - "opencv-python-headless>=4.8.0"
    - "numpy>=1.24.0"
    - "Pillow>=10.0.0"
  gpu: true

predict: "cog_predict.py:Predictor"
```

**Step 2: Write the Cog predictor**

```
training/cog_predict.py:
```
```python
"""Cog predictor for Replicate deployment."""

import json
import time

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from transformers import SegformerForSemanticSegmentation

# Inline config to avoid import issues in Cog container
NUM_CLASSES = 10
INPUT_SIZE = 512
COORD_RANGE = 1000

CLASS_KEY = {
    1: "wall", 2: "door", 3: "window", 4: "balcony",
    5: "balcony_window", 6: "bedroom", 7: "living_room",
    8: "kitchen", 9: "bathroom",
}

MIN_AREA = {
    "wall": 20, "door": 15, "window": 15, "balcony": 50,
    "balcony_window": 20, "bedroom": 100, "living_room": 100,
    "kitchen": 100, "bathroom": 50,
}

SIMPLIFY_EPSILON = {
    "wall": 1.5, "door": 2.0, "window": 2.0, "balcony": 2.0,
    "balcony_window": 2.0, "bedroom": 3.0, "living_room": 3.0,
    "kitchen": 3.0, "bathroom": 3.0,
}


def mask_to_polygons(mask, original_size, padding):
    """Convert segmentation mask to polygon JSON."""
    orig_h, orig_w = original_size
    pad_top, pad_left, content_h, content_w = padding

    result = {}
    for class_id, key in CLASS_KEY.items():
        binary = (mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        min_area = MIN_AREA.get(key, 20)
        epsilon = SIMPLIFY_EPSILON.get(key, 2.0)

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
            if len(simplified) < 3:
                continue
            points = simplified.squeeze().tolist()
            if len(points) < 3 or isinstance(points[0], (int, float)):
                continue

            normalized = []
            for px, py in points:
                x = (px - pad_left) / content_w * COORD_RANGE
                y = (py - pad_top) / content_h * COORD_RANGE
                normalized.append([
                    round(max(0, min(COORD_RANGE, x)), 1),
                    round(max(0, min(COORD_RANGE, y)), 1),
                ])
            if normalized[0] != normalized[-1]:
                normalized.append(normalized[0])
            polygons.append(normalized)

        result[key] = polygons
    return result


class Predictor(BasePredictor):
    def setup(self):
        """Load model into memory."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "./model", num_labels=NUM_CLASSES
        )
        self.model.to(self.device)

    def predict(
        self,
        image: Path = Input(description="Floorplan image"),
    ) -> str:
        """Run inference on a floorplan image."""
        start = time.time()

        # Preprocess
        img = Image.open(str(image)).convert("RGB")
        orig_w, orig_h = img.size
        scale = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        img_padded = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (255, 255, 255))
        img_padded.paste(img_resized, (0, 0))

        arr = np.array(img_padded).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(pixel_values=tensor)
            logits = torch.nn.functional.interpolate(
                outputs.logits, size=(INPUT_SIZE, INPUT_SIZE),
                mode="bilinear", align_corners=False,
            )
            mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Vectorize
        result = mask_to_polygons(
            mask,
            original_size=(orig_h, orig_w),
            padding=(0, 0, new_h, new_w),
        )
        result["version"] = 3
        result["image_width_meters"] = 0

        elapsed = time.time() - start
        result["_inference_time_ms"] = round(elapsed * 1000)

        return json.dumps(result)
```

**Step 3: Commit**

```bash
git add training/cog.yaml training/cog_predict.py
git commit -m "feat: add Cog container config for Replicate deployment"
```

---

### Task 8: Node.js integration - Replicate-based analyzer

**Files:**
- Create: `src/segmentation.js`
- Modify: `index.js:7,54`
- Modify: `.env.example`

**Step 1: Write the Replicate-based analyzer**

```
src/segmentation.js:
```
```javascript
import Replicate from 'replicate'

/**
 * Analyze a floor plan image using the segmentation model on Replicate.
 *
 * @param {Buffer} imageBuffer - raw image bytes
 * @param {string} mimeType - e.g. 'image/jpeg'
 * @returns {Promise<object>} parsed floor plan JSON with polygons
 */
export async function analyzeFloorplanSegmentation(imageBuffer, mimeType) {
  const apiToken = process.env.REPLICATE_API_TOKEN
  if (!apiToken) {
    throw new Error('REPLICATE_API_TOKEN environment variable is required')
  }

  const replicate = new Replicate({ auth: apiToken })

  // Convert buffer to data URI for Replicate
  const base64 = imageBuffer.toString('base64')
  const dataUri = `data:${mimeType};base64,${base64}`

  // Call the deployed model
  const modelVersion = process.env.SEGMENTATION_MODEL_VERSION
  if (!modelVersion) {
    throw new Error(
      'SEGMENTATION_MODEL_VERSION environment variable is required '
      + '(format: owner/model:version)'
    )
  }

  const output = await replicate.run(modelVersion, {
    input: { image: dataUri },
  })

  // Output is a JSON string from the Cog predictor
  const result = typeof output === 'string' ? JSON.parse(output) : output

  // Remove internal fields
  delete result._inference_time_ms

  return result
}
```

**Step 2: Update `index.js` to support both backends**

In `index.js`, add import after line 7:
```javascript
import { analyzeFloorplanSegmentation } from './src/segmentation.js'
```

Replace the try block in the `/analyze` route (lines 53-58):
```javascript
    try {
      const backend = process.env.ANALYSIS_BACKEND || 'gemini'
      let result
      if (backend === 'segmentation') {
        result = await analyzeFloorplanSegmentation(buffer, file.mimetype)
      } else {
        result = await analyzeFloorplan(buffer, file.mimetype)
      }
      return result
    } catch (err) {
      request.log.error({ err }, 'Floor plan analysis failed')
      return reply.code(502).send({ error: 'Floor plan analysis failed' })
    }
```

**Step 3: Update `.env.example`**

Add to `.env.example`:
```
ANALYSIS_BACKEND=gemini
SEGMENTATION_MODEL_VERSION=your-username/floorplan-segformer:version-hash
```

**Step 4: Commit**

```bash
git add src/segmentation.js index.js .env.example
git commit -m "feat: add segmentation model backend via Replicate"
```

---

### Task 9: Benchmark A9 - segmentation vs other approaches

**Files:**
- Create: `bench/a9-segmentation.js`

**Step 1: Write the benchmark**

```
bench/a9-segmentation.js:
```
```javascript
/**
 * A9: Segmentation model approach
 * Uses the deployed SegFormer model on Replicate for floorplan analysis.
 */

import 'dotenv/config'
import { loadTestImage, loadReference, printReport } from './utils.js'
import { analyzeFloorplanSegmentation } from '../src/segmentation.js'

async function run() {
  const imageBuffer = loadTestImage()
  const reference = loadReference()

  console.log('A9: Segmentation Model (SegFormer on Replicate)')
  console.log('Running inference...')

  const start = Date.now()
  const result = await analyzeFloorplanSegmentation(imageBuffer, 'image/jpeg')
  const elapsed = Date.now() - start

  printReport('A9: Segmentation Model', result, reference, elapsed)

  // Save result
  const fs = await import('fs')
  fs.writeFileSync('bench/a9-result.json', JSON.stringify(result, null, 2))
  console.log('\nResult saved to bench/a9-result.json')
}

run().catch(console.error)
```

**Step 2: Commit**

```bash
git add bench/a9-segmentation.js
git commit -m "feat: add A9 segmentation benchmark"
```

---

### Task 10: Training workflow documentation

**Files:**
- Create: `training/README.md`

**Step 1: Write the training README**

Create `training/README.md` with setup instructions, synthetic data generation commands, CubiCasa5K preparation steps, training commands, local testing commands, and Replicate deployment steps.

Cover:
- Python venv setup + `pip install -r requirements.txt`
- `python generate_synthetic.py --count 2000 --output-dir data/prepared/synthetic/train`
- `python generate_synthetic.py --count 500 --output-dir data/prepared/synthetic/val --seed 999`
- Optional: `git clone https://github.com/CubiCasa/CubiCasa5k data/cubicasa5k` + prepare_cubicasa.py
- `python train.py --data-dirs ... --val-dirs ... --output-dir checkpoints/segformer-floorplan`
- `python predict.py --model checkpoints/segformer-floorplan/best --image ../test-data/U3laAgYJ.jpg`
- Cog push to Replicate
- Class table (10 classes)

**Step 2: Commit**

```bash
git add training/README.md
git commit -m "docs: add training workflow documentation"
```
