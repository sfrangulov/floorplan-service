"""Local inference server for floorplan segmentation."""

import io
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler

from config import INPUT_SIZE, NUM_CLASSES
from vectorize import mask_to_polygons

MODEL_PATH = "checkpoints/segformer-floorplan-v2/best"
HOST = "0.0.0.0"
PORT = 5555


def load_model(model_path, device):
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_path, num_labels=NUM_CLASSES
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

    return {
        "version": 3,
        "image_width_meters": 0,
        "_inference_time_ms": elapsed,
        "elements": polygons,
    }


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
