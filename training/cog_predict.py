"""Cog predictor for floorplan segmentation model.

Self-contained predictor that inlines all constants and helpers
so the Cog container has no dependency on config.py or vectorize.py.
"""

import json
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cog import BasePredictor, Input, Path
from PIL import Image
from transformers import SegformerForSemanticSegmentation

# ---------------------------------------------------------------------------
# Inlined constants (must stay in sync with config.py / vectorize.py)
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
INPUT_SIZE = 512
COORD_RANGE = 1000

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

MIN_AREA = {
    "wall": 20,
    "door": 15,
    "window": 15,
    "balcony": 50,
    "balcony_window": 50,
    "bedroom": 100,
    "living_room": 100,
    "kitchen": 100,
    "bathroom": 100,
}

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


# ---------------------------------------------------------------------------
# Inlined mask_to_polygons (simplified version of vectorize.py)
# ---------------------------------------------------------------------------
def mask_to_polygons(
    mask: np.ndarray, original_size=None, padding=None
) -> dict:
    """Convert a segmentation mask to a dict of polygon lists.

    Args:
        mask: 2D uint8 array where pixel values are class IDs (0=background).
        original_size: (height, width) of the original image before any
            padding/resizing.  If None, uses the mask dimensions.
        padding: Optional tuple (pad_top, pad_left, content_h, content_w)
            describing how the original image was padded into the mask.

    Returns:
        Dict mapping class name -> list of polygons.
        Each polygon is a list of [x, y] points normalized to 0-COORD_RANGE.
        Polygons are closed (last point == first point).
    """
    mask_h, mask_w = mask.shape[:2]

    if original_size is None:
        original_size = (mask_h, mask_w)

    orig_h, orig_w = original_size

    if padding is not None:
        pad_top, pad_left, content_h, content_w = padding
        offset_x = pad_left
        offset_y = pad_top
        scale_x = orig_w / content_w
        scale_y = orig_h / content_h
    else:
        offset_x = 0
        offset_y = 0
        scale_x = orig_w / mask_w
        scale_y = orig_h / mask_h

    norm_x = COORD_RANGE / orig_w
    norm_y = COORD_RANGE / orig_h

    result = {name: [] for name in CLASS_KEY.values()}

    for class_id, class_name in CLASS_KEY.items():
        binary = ((mask == class_id) * 255).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = MIN_AREA[class_name]
        epsilon = SIMPLIFY_EPSILON[class_name]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            points = approx.reshape(-1, 2)

            polygon = []
            for px, py in points:
                ox = (px - offset_x) * scale_x
                oy = (py - offset_y) * scale_y

                nx = ox * norm_x
                ny = oy * norm_y

                nx = int(round(max(0, min(COORD_RANGE, nx))))
                ny = int(round(max(0, min(COORD_RANGE, ny))))

                polygon.append([nx, ny])

            if polygon and polygon[0] != polygon[-1]:
                polygon.append(polygon[0])

            result[class_name].append(polygon)

    return result


# ---------------------------------------------------------------------------
# Cog Predictor
# ---------------------------------------------------------------------------
class Predictor(BasePredictor):
    """Floorplan segmentation predictor for Replicate (Cog)."""

    def setup(self):
        """Load the model from the local ./model directory."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "./model",
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.train(False)

    def predict(
        self,
        image: Path = Input(description="Floorplan image to segment"),
    ) -> str:
        """Run segmentation on a floorplan image and return polygon JSON.

        Returns:
            JSON string with polygon data for each detected element class.
        """
        t0 = time.time()

        # 1. Open image and record original size
        pil_image = Image.open(str(image)).convert("RGB")
        orig_w, orig_h = pil_image.size

        # 2. Resize preserving aspect ratio, pad to INPUT_SIZE x INPUT_SIZE
        scale = INPUT_SIZE / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pil_resized = pil_image.resize((new_w, new_h), Image.BILINEAR)

        # Create white canvas and paste resized image (centered)
        canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (255, 255, 255))
        pad_left = (INPUT_SIZE - new_w) // 2
        pad_top = (INPUT_SIZE - new_h) // 2
        canvas.paste(pil_resized, (pad_left, pad_top))

        # 3. Convert to tensor (CHW, float32, 0-1)
        img_np = np.array(canvas, dtype=np.float32) / 255.0
        img_tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        # 4. Run inference
        with torch.no_grad():
            outputs = self.model(pixel_values=img_tensor)
            logits = outputs.logits  # (1, C, H/4, W/4)

        # 5. Interpolate logits to INPUT_SIZE x INPUT_SIZE, take argmax
        logits = F.interpolate(
            logits,
            size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # 6. Convert mask to polygons
        padding = (pad_top, pad_left, new_h, new_w)
        polygons = mask_to_polygons(
            mask, original_size=(orig_h, orig_w), padding=padding
        )

        inference_ms = round((time.time() - t0) * 1000, 1)

        # 7. Build result
        result = {
            "version": 3,
            "image_width_meters": 0,
            "_inference_time_ms": inference_ms,
            **polygons,
        }

        return json.dumps(result)
