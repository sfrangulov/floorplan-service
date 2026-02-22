"""Local inference server for floorplan segmentation."""

import io
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
from scipy import ndimage

from config import INPUT_SIZE, NUM_CLASSES, COORD_RANGE
from vectorize import mask_to_polygons

MODEL_PATH = "checkpoints/segformer-floorplan-v3/best"
HOST = "0.0.0.0"
PORT = 5555


def load_model(model_path, device):
    from pathlib import Path
    from transformers import SegformerForSemanticSegmentation

    local_path = str(Path(model_path).resolve())
    model = SegformerForSemanticSegmentation.from_pretrained(
        local_path, num_labels=NUM_CLASSES
    )
    model.to(device)
    model.eval()
    return model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_inference(model, image_bytes, device):
    t0 = time.time()

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = pil_image.size

    scale = INPUT_SIZE / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pil_resized = pil_image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (255, 255, 255))
    pad_left = (INPUT_SIZE - new_w) // 2
    pad_top = (INPUT_SIZE - new_h) // 2
    canvas.paste(pil_resized, (pad_left, pad_top))

    img_np = np.array(canvas, dtype=np.float32) / 255.0
    img_tensor = (
        torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        outputs = model(pixel_values=img_tensor)
        logits = F.interpolate(
            outputs.logits,
            size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    padding = (pad_top, pad_left, new_h, new_w)
    polygons = mask_to_polygons(
        mask, original_size=(orig_h, orig_w), padding=padding
    )

    elapsed = round((time.time() - t0) * 1000, 1)

    # Compute apartments boundary from non-background mask
    non_bg = ((mask > 0) * 255).astype(np.uint8)
    # Remove padding region
    if padding is not None:
        pad_top, pad_left, content_h, content_w = padding
        non_bg_cropped = non_bg[pad_top:pad_top + content_h, pad_left:pad_left + content_w]
    else:
        non_bg_cropped = non_bg
    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    non_bg_closed = cv2.morphologyEx(non_bg_cropped, cv2.MORPH_CLOSE, kernel)
    apt_contours, _ = cv2.findContours(non_bg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Estimate pixels_per_meter: assume longest apartment dimension ~ 15m
    pixels_per_meter = max(orig_w, orig_h) / 15.0

    # Convert from normalized 0-COORD_RANGE to pixel coordinates
    # and flatten into target v2 format
    result = {
        "version": 2,
        "width": orig_w,
        "height": orig_h,
        "pixels_per_meter": round(pixels_per_meter, 2),
        "_inference_time_ms": elapsed,
    }

    scale_x = orig_w / COORD_RANGE
    scale_y = orig_h / COORD_RANGE

    for class_name, poly_list in polygons.items():
        pixel_polys = []
        for poly in poly_list:
            pixel_poly = [
                [round(pt[0] * scale_x, 2), round(pt[1] * scale_y, 2)]
                for pt in poly
            ]
            pixel_polys.append(pixel_poly)
        result[class_name] = pixel_polys

    # Compute apartments polygon from largest non-background contour
    if apt_contours:
        largest = max(apt_contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(largest, 3.0, closed=True)
        pts = approx.reshape(-1, 2)
        # Scale from cropped mask space to pixel coordinates
        crop_scale_x = orig_w / non_bg_cropped.shape[1]
        crop_scale_y = orig_h / non_bg_cropped.shape[0]
        apt_poly = [
            [round(float(px) * crop_scale_x, 2), round(float(py) * crop_scale_y, 2)]
            for px, py in pts
        ]
        if apt_poly and apt_poly[0] != apt_poly[-1]:
            apt_poly.append(apt_poly[0])
        result["apartments"] = apt_poly
    else:
        result["apartments"] = []

    # Add missing classes expected by the format
    for key in ("other_room", "kitchen_table",
                "kitchen_zone", "sink", "cooker"):
        if key not in result:
            result[key] = []

    return result


class Handler(BaseHTTPRequestHandler):
    model = None
    device = None

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        image_bytes = self.rfile.read(content_length)

        result = run_inference(self.model, image_bytes, self.device)

        response = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        print(f"[serve] {args[0]}")


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, device)
    print("Model loaded.")

    Handler.model = model
    Handler.device = device

    server = HTTPServer((HOST, PORT), Handler)
    print(f"Serving on http://{HOST}:{PORT}/predict")
    server.serve_forever()


if __name__ == "__main__":
    main()
