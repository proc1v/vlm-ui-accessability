#!/usr/bin/env python3
"""Gradio app for side-by-side GT vs Predicted AX tree comparison."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from utils.parser import AXNode, parse_tree, tree_to_text, is_leaf
from utils.normalize import denormalize_bbox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

CATEGORY_COLORS_FILL = config.CATEGORY_COLORS
CATEGORY_COLORS_BORDER = config.CATEGORY_BORDER_COLORS

# Map each role to its category
ROLE_TO_CATEGORY = config.ROLE_TO_CATEGORY


def get_colors(
    role: str,
    role_to_cat: Optional[Dict[str, str]] = None,
    fill_colors: Optional[Dict[str, Tuple]] = None,
    border_colors: Optional[Dict[str, Tuple]] = None,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return (fill_rgba, border_rgba) for a role."""
    r2c = role_to_cat or ROLE_TO_CATEGORY
    fc = fill_colors or CATEGORY_COLORS_FILL
    bc = border_colors or CATEGORY_COLORS_BORDER
    cat = r2c.get(role, "Other")
    fill = fc.get(cat, (121, 85, 72, 80))
    border = bc.get(cat, (121, 85, 72, 255))
    return fill, border


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_boxes_on_image(
    image: Image.Image,
    nodes: List[AXNode],
    visible_roles: Set[str],
    role_to_cat: Optional[Dict[str, str]] = None,
    fill_colors: Optional[Dict[str, Tuple]] = None,
    border_colors: Optional[Dict[str, Tuple]] = None,
) -> Image.Image:
    """Draw bounding boxes on a copy of the image.

    Containers are drawn first (lower z-order), then leaves on top.
    """
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    img_w, img_h = img.size

    # Try to load a small font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 10)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Separate containers and leaves for z-ordering
    container_nodes = []
    leaf_nodes = []
    for node in nodes:
        if node.role not in visible_roles:
            continue
        if node.bbox is None:
            continue
        if is_leaf(node):
            leaf_nodes.append(node)
        else:
            container_nodes.append(node)

    # Draw containers first, then leaves
    for node in container_nodes + leaf_nodes:
        bbox_px = denormalize_bbox(node.bbox, img_w, img_h, source_range=1000)
        x1, y1, x2, y2 = bbox_px

        # Clamp to image bounds
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        fill, border = get_colors(node.role, role_to_cat, fill_colors, border_colors)

        # Semi-transparent filled rectangle
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=border, width=1)

        # Label
        label = node.role
        if node.subrole:
            label += f"({node.subrole})"

        # Draw label background
        try:
            text_bbox = font.getbbox(label)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)

        label_y = max(0, y1 - th - 2)
        draw.rectangle(
            [x1, label_y, x1 + tw + 4, label_y + th + 2],
            fill=(border[0], border[1], border[2], 200),
        )
        draw.text((x1 + 2, label_y), label, fill=(255, 255, 255, 255), font=font)

    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _numeric_sort_key(s: str):
    """Sort key that orders numeric stems by value (0, 1, 2, … 10, 11, …)."""
    try:
        return (0, int(s))
    except ValueError:
        return (1, s)


class DataManager:
    """Manages dataset, predictions, and metrics for the Gradio app."""

    def __init__(
        self,
        images_dir: Path,
        gt_dir: Path,
        pred_dir: Path,
        metrics_file: Optional[Path] = None,
    ):
        self.images_dir = images_dir
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir

        # Discover samples (intersection of images and GT)
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        image_stems = {
            f.stem
            for f in images_dir.iterdir()
            if f.suffix.lower() in image_exts
        }
        gt_stems = {f.stem for f in gt_dir.glob("*.txt")}
        self.stems = sorted(image_stems & gt_stems, key=_numeric_sort_key)

        if not self.stems:
            logger.warning("No matching samples found between images and GT.")

        # Load per-sample metrics if available
        self.metrics: Dict[str, Dict] = {}
        if metrics_file and metrics_file.exists():
            try:
                with open(metrics_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if "filename" in row:
                            self.metrics[row["filename"]] = row
                logger.info(f"Loaded metrics for {len(self.metrics)} samples")
            except Exception as e:
                logger.warning(f"Could not load metrics file: {e}")

        # Find image extension for each stem
        self._image_paths: Dict[str, Path] = {}
        for stem in self.stems:
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                p = images_dir / f"{stem}{ext}"
                if p.exists():
                    self._image_paths[stem] = p
                    break

    def get_sample_list(self) -> List[str]:
        return self.stems

    def get_image(self, stem: str) -> Optional[Image.Image]:
        path = self._image_paths.get(stem)
        if path and path.exists():
            return Image.open(path)
        return None

    def get_gt_text(self, stem: str) -> str:
        path = self.gt_dir / f"{stem}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def get_pred_text(self, stem: str) -> str:
        path = self.pred_dir / f"{stem}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def get_metrics(self, stem: str) -> Dict:
        return self.metrics.get(stem, {})

    def get_sorted_stems(self, sort_by: str) -> List[str]:
        """Return stems sorted by the given criterion."""
        if sort_by == "filename":
            return sorted(self.stems, key=_numeric_sort_key)
        elif sort_by == "Edge F1 (ascending)":
            return sorted(
                self.stems,
                key=lambda s: self.metrics.get(s, {}).get("edge_f1", 0.0),
            )
        elif sort_by == "GED (descending)":
            return sorted(
                self.stems,
                key=lambda s: self.metrics.get(s, {}).get("ged", 0.0) or 0.0,
                reverse=True,
            )
        elif sort_by == "Node count":
            return sorted(
                self.stems,
                key=lambda s: self.metrics.get(s, {}).get("gt_node_count", 0),
                reverse=True,
            )
        return self.stems


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------


def build_app(data: DataManager, simplified: bool = False) -> gr.Blocks:
    """Build the Gradio visualization app."""

    if simplified:
        role_categories = config.SIMPLIFIED_ROLE_CATEGORIES
        role_to_cat = config.SIMPLIFIED_ROLE_TO_CATEGORY
        fill_colors = config.SIMPLIFIED_CATEGORY_COLORS
        border_colors_map = config.SIMPLIFIED_CATEGORY_BORDER_COLORS
        role_mapping = config.SIMPLIFIED_ROLE_MAPPING
    else:
        role_categories = config.ROLE_CATEGORIES
        role_to_cat = config.ROLE_TO_CATEGORY
        fill_colors = config.CATEGORY_COLORS
        border_colors_map = config.CATEGORY_BORDER_COLORS
        role_mapping = None

    all_roles = config.SIMPLIFIED_ROLES if simplified else config.VALID_ROLES

    with gr.Blocks(title="Screen2AX Visualization", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Screen2AX: GT vs Prediction Comparison")

        # -- Top bar: sample selector --
        with gr.Row():
            sort_dropdown = gr.Dropdown(
                choices=["filename", "Edge F1 (ascending)", "GED (descending)", "Node count"],
                value="filename",
                label="Sort by",
                scale=1,
            )
            prev_btn = gr.Button("\u25C0 Prev", size="sm", scale=0, min_width=80)
            sample_dropdown = gr.Dropdown(
                choices=data.get_sorted_stems("filename"),
                value=data.stems[0] if data.stems else None,
                label="Sample",
                scale=3,
            )
            next_btn = gr.Button("Next \u25B6", size="sm", scale=0, min_width=80)
            metrics_display = gr.Textbox(
                label="Per-sample Metrics",
                interactive=False,
                scale=2,
            )

        # -- Middle section: side by side images --
        with gr.Row():
            gt_image = gr.Image(label="Ground Truth", type="pil")
            pred_image = gr.Image(label="Prediction", type="pil")

        # -- Bottom: class filter --
        gr.Markdown("### Role Filter")

        with gr.Row():
            select_all_btn = gr.Button("Select All", size="sm")
            deselect_all_btn = gr.Button("Deselect All", size="sm")

        role_checkboxes = {}
        cat_toggle_btns = {}
        with gr.Row():
            for cat_name, cat_roles in role_categories.items():
                with gr.Column(min_width=150):
                    cat_toggle_btns[cat_name] = gr.Button(
                        f"Toggle {cat_name}", size="sm", variant="secondary",
                    )
                    role_checkboxes[cat_name] = gr.CheckboxGroup(
                        choices=cat_roles,
                        value=cat_roles,  # all selected by default
                        label=cat_name,
                        show_label=False,
                    )

        # -- Side panel: tree text --
        gr.Markdown("### Tree Text")
        with gr.Row():
            gt_text_box = gr.Textbox(
                label="GT Tree",
                lines=20,
                max_lines=40,
                interactive=False,
            )
            pred_text_box = gr.Textbox(
                label="Predicted Tree",
                lines=20,
                max_lines=40,
                interactive=False,
            )

        # -- State: current visible roles --
        # We'll collect all checkboxes into one set for drawing

        def get_visible_roles(*checkbox_values) -> Set[str]:
            """Collect all checked roles from all category checkbox groups."""
            visible = set()
            for val_list in checkbox_values:
                if val_list:
                    visible.update(val_list)
            return visible

        def update_display(
            sample_stem: str,
            *checkbox_values,
        ):
            """Update images and text when sample or filters change."""
            if not sample_stem:
                return None, None, "", "", ""

            visible = get_visible_roles(*checkbox_values)

            image = data.get_image(sample_stem)
            gt_text = data.get_gt_text(sample_stem)
            pred_text = data.get_pred_text(sample_stem)
            metrics = data.get_metrics(sample_stem)

            gt_nodes = parse_tree(gt_text, role_mapping=role_mapping)
            pred_nodes = parse_tree(pred_text, role_mapping=role_mapping)

            gt_img = draw_boxes_on_image(image, gt_nodes, visible, role_to_cat, fill_colors, border_colors_map) if image else None
            pred_img = draw_boxes_on_image(image, pred_nodes, visible, role_to_cat, fill_colors, border_colors_map) if image else None

            # Format metrics
            parts = []
            if metrics:
                for k in ["edge_f1", "leaves_f1", "complete_match", "ged"]:
                    v = metrics.get(k)
                    if v is not None:
                        if isinstance(v, float):
                            parts.append(f"{k}: {v:.3f}")
                        else:
                            parts.append(f"{k}: {v}")
                parts.append(f"GT nodes: {metrics.get('gt_node_count', '?')}")
                parts.append(f"Pred nodes: {metrics.get('pred_node_count', '?')}")
            metrics_str = " | ".join(parts) if parts else "N/A"

            return gt_img, pred_img, gt_text, pred_text, metrics_str

        # Gather all checkbox components in category order
        checkbox_components = [
            role_checkboxes[cat] for cat in role_categories
        ]

        all_inputs = [sample_dropdown] + checkbox_components
        all_outputs = [gt_image, pred_image, gt_text_box, pred_text_box, metrics_display]

        # Sample change triggers update
        sample_dropdown.change(
            fn=update_display,
            inputs=all_inputs,
            outputs=all_outputs,
        )

        # Each checkbox group change triggers update
        for cb in checkbox_components:
            cb.change(
                fn=update_display,
                inputs=all_inputs,
                outputs=all_outputs,
            )

        # Sort dropdown changes the sample list
        def update_sort(sort_by: str):
            stems = data.get_sorted_stems(sort_by)
            first = stems[0] if stems else None
            return gr.update(choices=stems, value=first)

        sort_dropdown.change(
            fn=update_sort,
            inputs=[sort_dropdown],
            outputs=[sample_dropdown],
        )

        # Select all / Deselect all
        def select_all():
            return [gr.update(value=roles) for roles in role_categories.values()]

        def deselect_all():
            return [gr.update(value=[]) for _ in role_categories]

        select_all_btn.click(fn=select_all, outputs=checkbox_components)
        deselect_all_btn.click(fn=deselect_all, outputs=checkbox_components)

        # Per-category toggle buttons
        for cat_name in role_categories:
            cat_roles = role_categories[cat_name]
            cb = role_checkboxes[cat_name]

            def make_toggle(roles):
                def toggle(current):
                    return gr.update(value=[] if current else roles)
                return toggle

            cat_toggle_btns[cat_name].click(
                fn=make_toggle(cat_roles),
                inputs=[cb],
                outputs=[cb],
            )

        # Prev / Next navigation
        def go_prev(current_stem, sort_by):
            stems = data.get_sorted_stems(sort_by)
            if not stems:
                return gr.update()
            try:
                idx = stems.index(current_stem)
            except ValueError:
                idx = 0
            new_idx = (idx - 1) % len(stems)
            return gr.update(value=stems[new_idx])

        def go_next(current_stem, sort_by):
            stems = data.get_sorted_stems(sort_by)
            if not stems:
                return gr.update()
            try:
                idx = stems.index(current_stem)
            except ValueError:
                idx = 0
            new_idx = (idx + 1) % len(stems)
            return gr.update(value=stems[new_idx])

        prev_btn.click(
            fn=go_prev,
            inputs=[sample_dropdown, sort_dropdown],
            outputs=[sample_dropdown],
        )
        next_btn.click(
            fn=go_next,
            inputs=[sample_dropdown, sort_dropdown],
            outputs=[sample_dropdown],
        )

        # Initial load
        app.load(
            fn=update_display,
            inputs=all_inputs,
            outputs=all_outputs,
        )

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gradio visualization for Screen2AX GT vs Prediction comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing screenshot images",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        required=True,
        help="Directory with GT annotation .txt files",
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Directory with predicted .txt files",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Path to per-sample metrics JSONL file (optional)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run Gradio server on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--simplified-roles",
        action="store_true",
        help="Use 7-class simplified role mapping from Screen2AX paper",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    data = DataManager(
        images_dir=Path(args.images_dir),
        gt_dir=Path(args.gt_dir),
        pred_dir=Path(args.pred_dir),
        metrics_file=Path(args.metrics_file) if args.metrics_file else None,
    )

    if not data.stems:
        logger.error(
            "No matching samples found. Check that images-dir, gt-dir, and pred-dir "
            "share matching filenames."
        )
        sys.exit(1)

    logger.info(f"Loaded {len(data.stems)} samples")

    app = build_app(data, simplified=args.simplified_roles)
    app.launch(server_port=args.port, share=args.share)
