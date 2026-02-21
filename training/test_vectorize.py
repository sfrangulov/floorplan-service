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


def test_all_class_keys_present():
    """Result dict should contain all expected class keys."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = mask_to_polygons(mask, original_size=(100, 100))
    expected_keys = {"wall", "door", "window", "balcony", "balcony_window",
                     "bedroom", "living_room", "kitchen", "bathroom"}
    assert set(result.keys()) == expected_keys


def test_small_contours_filtered():
    """Contours smaller than MIN_AREA should be filtered out."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Single tiny pixel - area ~1, well below any MIN_AREA threshold
    mask[50, 50] = 1  # wall class, 1 pixel
    result = mask_to_polygons(mask, original_size=(200, 200))
    assert result["wall"] == []


def test_polygons_are_closed():
    """Each polygon's last point should equal its first point (closed)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 8  # kitchen
    result = mask_to_polygons(mask, original_size=(100, 100))
    assert len(result["kitchen"]) == 1
    poly = result["kitchen"][0]
    assert poly[0] == poly[-1], "Polygon should be closed (first == last)"


def test_padding_offset():
    """When padding is provided, coordinates should be adjusted."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Place a wall in the content area which starts at (10, 20)
    # Content area is 80x60 within the 100x100 padded mask
    mask[15:20, 25:55] = 1  # wall inside content area
    padding = (10, 20, 80, 60)  # pad_top, pad_left, content_h, content_w
    result = mask_to_polygons(mask, original_size=(80, 60), padding=padding)
    assert len(result["wall"]) == 1
    for x, y in result["wall"][0]:
        assert 0 <= x <= 1000, f"x={x} out of range after padding adjustment"
        assert 0 <= y <= 1000, f"y={y} out of range after padding adjustment"


def test_multiple_contours_same_class():
    """Multiple separate regions of the same class should produce multiple polygons."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[10:30, 10:30] = 6   # bedroom top-left
    mask[100:130, 100:130] = 6  # bedroom bottom-right
    result = mask_to_polygons(mask, original_size=(200, 200))
    assert len(result["bedroom"]) == 2


def test_door_and_window():
    """Doors and windows should be recognized."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:50, 10:25] = 2  # door
    mask[40:50, 60:75] = 3  # window
    result = mask_to_polygons(mask, original_size=(100, 100))
    assert len(result["door"]) == 1
    assert len(result["window"]) == 1
