"""Coordinate normalization utilities for Screen2AX bounding boxes."""

from __future__ import annotations

from typing import List


def normalize_bbox(
    bbox: List[int],
    img_width: int,
    img_height: int,
    target_range: int = 1000,
) -> List[int]:
    """Convert pixel coordinates [x1,y1,x2,y2] to normalized [0, target_range]."""
    x1, y1, x2, y2 = bbox
    return [
        int(x1 / img_width * target_range),
        int(y1 / img_height * target_range),
        int(x2 / img_width * target_range),
        int(y2 / img_height * target_range),
    ]


def denormalize_bbox(
    bbox: List[int],
    img_width: int,
    img_height: int,
    source_range: int = 1000,
) -> List[int]:
    """Convert normalized coordinates back to pixel coordinates."""
    x1, y1, x2, y2 = bbox
    return [
        int(x1 / source_range * img_width),
        int(y1 / source_range * img_height),
        int(x2 / source_range * img_width),
        int(y2 / source_range * img_height),
    ]


def xywh_to_xyxy(bbox: List[int]) -> List[int]:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_xywh(bbox: List[int]) -> List[int]:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]
