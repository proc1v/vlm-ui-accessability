"""Screen2AX evaluation utilities."""

from .parser import AXNode, parse_line, parse_tree, tree_to_text
from .normalize import (
    normalize_bbox,
    denormalize_bbox,
    xywh_to_xyxy,
    xyxy_to_xywh,
)
from .metrics import evaluate_sample, evaluate_batch
