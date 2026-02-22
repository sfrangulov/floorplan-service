"""Convert segmentation mask to polygon JSON."""
import cv2
import numpy as np

from config import COORD_RANGE

# Class ID -> class name mapping
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

# Minimum contour area (in mask pixels) to keep per class
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

# Douglas-Peucker simplification epsilon per class
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


def mask_to_polygons(mask: np.ndarray, original_size=None, padding=None) -> dict:
    """Convert a segmentation mask to a dict of polygon lists.

    Args:
        mask: 2D uint8 array where pixel values are class IDs (0=background).
        original_size: (height, width) of the original image before any
            padding/resizing. Used for coordinate scaling. If None, uses
            the mask dimensions.
        padding: Optional tuple (pad_top, pad_left, content_h, content_w)
            describing how the original image was padded into the mask.
            When provided, contour coordinates are first shifted by the
            padding offset, then scaled relative to content dimensions.

    Returns:
        Dict mapping class name -> list of polygons.
        Each polygon is a list of [x, y] points normalized to 0-COORD_RANGE.
        Polygons are closed (last point == first point).
    """
    mask_h, mask_w = mask.shape[:2]

    if original_size is None:
        original_size = (mask_h, mask_w)

    orig_h, orig_w = original_size

    # Determine how to transform contour coordinates to original image space
    if padding is not None:
        pad_top, pad_left, content_h, content_w = padding
        # Contour coords are in mask space; shift by padding offset
        offset_x = pad_left
        offset_y = pad_top
        # Scale factor: content region in mask -> original image
        scale_x = orig_w / content_w
        scale_y = orig_h / content_h
    else:
        offset_x = 0
        offset_y = 0
        scale_x = orig_w / mask_w
        scale_y = orig_h / mask_h

    # Normalization factor: original image coords -> 0-COORD_RANGE
    norm_x = COORD_RANGE / orig_w
    norm_y = COORD_RANGE / orig_h

    result = {name: [] for name in CLASS_KEY.values()}

    # Linear elements (walls, doors, windows) need morphological splitting
    # to break connected grids into individual segments
    LINEAR_CLASSES = {"wall", "door", "window", "balcony_window"}

    for class_id, class_name in CLASS_KEY.items():
        # 1. Extract binary mask for this class
        binary = ((mask == class_id) * 255).astype(np.uint8)

        # 1b. For walls, erode to break the connected grid at junctions,
        # then dilate back to restore segment thickness
        if class_name == "wall":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.erode(binary, kernel, iterations=2)
            binary = cv2.dilate(binary, kernel, iterations=2)

        # 2. Find external contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = MIN_AREA[class_name]
        epsilon = SIMPLIFY_EPSILON[class_name]

        for contour in contours:
            # 3. Filter by minimum area
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # 3b. Door/window shape validation: filter square-ish blobs
            if class_name in ("door", "window"):
                _, (bw, bh), _ = cv2.minAreaRect(contour)
                if bw > 0 and bh > 0:
                    aspect = max(bw, bh) / min(bw, bh)
                    if aspect < 1.8 and area > 150:
                        continue

            # 4. Simplify with Douglas-Peucker
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)

            # Convert from OpenCV shape (N, 1, 2) to list of [x, y]
            points = approx.reshape(-1, 2)

            # 5. Transform coordinates
            #    a) Remove padding offset
            #    b) Scale to original image space
            #    c) Normalize to 0-COORD_RANGE
            polygon = []
            for px, py in points:
                # Mask coords -> original image coords
                ox = (px - offset_x) * scale_x
                oy = (py - offset_y) * scale_y

                # Original coords -> normalized 0-COORD_RANGE
                nx = ox * norm_x
                ny = oy * norm_y

                # Clamp to valid range
                nx = int(round(max(0, min(COORD_RANGE, nx))))
                ny = int(round(max(0, min(COORD_RANGE, ny))))

                polygon.append([nx, ny])

            # 6. Close polygon if not already closed
            if polygon and polygon[0] != polygon[-1]:
                polygon.append(polygon[0])

            result[class_name].append(polygon)

    return result
