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
    "wall": 2.5,
    "door": 2.0,
    "window": 2.0,
    "balcony": 2.0,
    "balcony_window": 2.0,
    "bedroom": 3.0,
    "living_room": 3.0,
    "kitchen": 3.0,
    "bathroom": 3.0,
}


def _straighten_polygon(points: np.ndarray, **_kwargs) -> np.ndarray:
    """Make wall polygon fully rectilinear (axis-aligned edges only).

    1. Classify each edge as horizontal or vertical.
    2. For each vertex, snap coordinates based on adjacent edge types:
       - From the incoming H edge: inherit y from the previous vertex.
       - From the incoming V edge: inherit x from the previous vertex.
    """
    pts = points.astype(np.float64).copy()
    n = len(pts)
    if n < 3:
        return pts

    # Step 1: classify each edge as H or V based on original geometry
    edge_type = []  # 'H' or 'V' for edge i→(i+1)
    for i in range(n):
        j = (i + 1) % n
        dx = abs(pts[j][0] - pts[i][0])
        dy = abs(pts[j][1] - pts[i][1])
        edge_type.append("H" if dx >= dy else "V")

    # Step 2: snap vertices — walk the polygon, propagating coordinates
    # For an H edge (i→j): j gets y from i
    # For a V edge (i→j): j gets x from i
    snapped = pts.copy()
    for i in range(n):
        j = (i + 1) % n
        if edge_type[i] == "H":
            snapped[j][1] = snapped[i][1]
        else:
            snapped[j][0] = snapped[i][0]

    # Step 3: second pass in reverse to fix accumulated drift
    for i in range(n - 1, -1, -1):
        j = (i + 1) % n
        if edge_type[i] == "H":
            avg_y = (snapped[i][1] + snapped[j][1]) / 2
            snapped[i][1] = avg_y
            snapped[j][1] = avg_y
        else:
            avg_x = (snapped[i][0] + snapped[j][0]) / 2
            snapped[i][0] = avg_x
            snapped[j][0] = avg_x

    # Remove consecutive near-duplicates
    cleaned = [snapped[0]]
    for i in range(1, n):
        if not (abs(snapped[i][0] - cleaned[-1][0]) < 1.0
                and abs(snapped[i][1] - cleaned[-1][1]) < 1.0):
            cleaned.append(snapped[i])
    return np.array(cleaned)


def _snap_door_to_wall(rect_center, wall_angle, wall_xs, wall_ys):
    """Snap door center onto the wall's center line.

    For a horizontal wall, aligns door center Y to the wall's mean Y.
    For a vertical wall, aligns door center X to the wall's mean X.
    """
    cx, cy = rect_center
    if wall_angle == 0.0:
        # Horizontal wall: snap Y to wall center
        cy = float(np.mean(wall_ys))
    else:
        # Vertical wall: snap X to wall center
        cx = float(np.mean(wall_xs))
    return (cx, cy)


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
        # then dilate back with extra iteration to close junction gaps
        if class_name == "wall":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.erode(binary, kernel, iterations=2)
            binary = cv2.dilate(binary, kernel, iterations=3)

        # 2. Find external contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = MIN_AREA[class_name]
        epsilon = SIMPLIFY_EPSILON[class_name]

        for contour in contours:
            # 3. Filter by minimum area
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # 3b. Window shape validation: filter square-ish blobs
            if class_name == "window":
                _, (bw, bh), _ = cv2.minAreaRect(contour)
                if bw > 0 and bh > 0:
                    aspect = max(bw, bh) / min(bw, bh)
                    if aspect < 1.8 and area > 150:
                        continue

            # 3c. Doors: convert blob to thin "closed door" rectangle
            #     Orient along adjacent wall using PCA on nearby wall pixels
            if class_name == "door":
                rect_center, rect_size, _ = cv2.minAreaRect(contour)
                door_length = max(rect_size)
                door_thickness = 5.0

                # Find wall pixels adjacent to this door
                door_only = np.zeros_like(binary)
                cv2.drawContours(door_only, [contour], -1, 255, thickness=cv2.FILLED)
                adj_kernel = np.ones((9, 9), np.uint8)
                dilated = cv2.dilate(door_only, adj_kernel)
                wall_nearby = ((mask == 1) * 255).astype(np.uint8)
                adjacent_wall = dilated & wall_nearby
                ys, xs = np.where(adjacent_wall > 0)

                if len(xs) >= 3:
                    # Determine wall direction: horizontal or vertical
                    x_spread = float(np.ptp(xs))
                    y_spread = float(np.ptp(ys))
                    wall_angle = 0.0 if x_spread >= y_spread else 90.0
                else:
                    # Fallback: use bounding rect aspect
                    _, _, bw, bh = cv2.boundingRect(contour)
                    wall_angle = 0.0 if bw >= bh else 90.0

                # Snap door center onto the wall's center line
                if len(xs) >= 3:
                    rect_center = _snap_door_to_wall(
                        rect_center, wall_angle, xs, ys
                    )

                thin_rect = (rect_center, (door_length, door_thickness), wall_angle)
                points = cv2.boxPoints(thin_rect).astype(np.float32)
            else:
                # 4. Simplify with Douglas-Peucker
                approx = cv2.approxPolyDP(contour, epsilon, closed=True)
                points = approx.reshape(-1, 2)

                # 4b. Straighten wall polygons: snap all edges to H/V
                if class_name == "wall":
                    points = _straighten_polygon(points, angle_threshold=45.0)

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
