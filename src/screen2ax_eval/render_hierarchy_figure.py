"""Render the two-panel accessibility-hierarchy figure for a screenshot.

Left panel  — the screenshot with bounding boxes colored by tree depth
               (root / top container / mid group / leaf).
Right panel — the corresponding tree: circles per node, edges = parent/child.

Produces a PNG (and optionally an SVG placeholder) suitable for inclusion in
a thesis. Can render one tree or a triptych (e.g. GT vs LoRA vs YOLO) sharing
the same screenshot on the left.

Usage (single tree):
    python render_hierarchy_figure.py \\
        --image data/val_images_symlinked/1000.png \\
        --tree  screen2ax_eval/results/qwen3vl_230b_simple/parsed/1000.txt \\
        --out   figures/1000_hierarchy.png

Usage (triptych — same screenshot, 3 trees side by side):
    python render_hierarchy_figure.py \\
        --image data/val_images_symlinked/1000.png \\
        --tree  silver=results/qwen3vl_230b_simple/parsed/1000.txt \\
                lora=inference_results_qwen_lora/val/parsed/1000.txt \\
                yolo=inference_results_yolo_composed/.../val/parsed/1000.txt \\
        --out   figures/1000_compare.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.parser import AXNode, parse_tree  # noqa: E402

logger = logging.getLogger("render_hierarchy")


# ---------------------------------------------------------------------------
# Colour scheme (matches the Screen2AX paper figure)

COLOR_ROOT = (230, 120, 45)     # orange — root window
COLOR_TOP  = (55, 120, 215)     # blue   — top-level container
COLOR_MID  = (240, 190, 60)     # yellow — mid-level group
COLOR_LEAF = (140, 100, 200)    # purple — leaves

BG_COLOR     = (255, 255, 255)
EDGE_COLOR   = (60, 60, 60)
TEXT_COLOR   = (20, 20, 20)


# ---------------------------------------------------------------------------
# Classification

def classify_depth(node: AXNode, tree_max_depth: int) -> str:
    """Bucket a node into root / top / mid / leaf."""
    if not node.children:
        return "leaf"
    if node.depth == 0:
        return "root"
    if node.depth == 1:
        return "top"
    # Everything between depth 2 and max-1 is "mid"
    return "mid"


BUCKET_COLOR = {
    "root": COLOR_ROOT,
    "top":  COLOR_TOP,
    "mid":  COLOR_MID,
    "leaf": COLOR_LEAF,
}
BUCKET_RADIUS = {"root": 26, "top": 20, "mid": 16, "leaf": 11}


# ---------------------------------------------------------------------------
# Left panel: bbox overlay


def render_bboxes_on_image(
    image: Image.Image,
    nodes: List[AXNode],
    max_depth: int,
    line_width: int = 3,
    alpha: int = 220,
) -> Image.Image:
    """Return a copy of image with node bboxes drawn, colored by bucket."""
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    W, H = img.size

    # Containers first (lower z-order), leaves on top.
    containers, leaves = [], []
    for n in nodes:
        if n.bbox is None:
            continue
        (leaves if not n.children else containers).append(n)

    # Containers deepest last so deeper outlines sit above ancestors.
    containers.sort(key=lambda n: n.depth)

    for node in containers + leaves:
        x1, y1, x2, y2 = node.bbox
        px1 = max(0, min(W - 1, int(round(x1 / 1000 * W))))
        py1 = max(0, min(H - 1, int(round(y1 / 1000 * H))))
        px2 = max(0, min(W - 1, int(round(x2 / 1000 * W))))
        py2 = max(0, min(H - 1, int(round(y2 / 1000 * H))))
        if px2 <= px1 or py2 <= py1:
            continue
        bucket = classify_depth(node, max_depth)
        color = BUCKET_COLOR[bucket] + (alpha,)
        draw.rectangle([px1, py1, px2, py2], outline=color, width=line_width)

    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Right panel: tree layout (pure-Python layered layout)


@dataclass
class LayoutNode:
    ax: AXNode
    bucket: str
    depth: int
    x: float = 0.0   # set by layout
    y: float = 0.0
    children: List["LayoutNode"] = None


def build_layout(nodes: List[AXNode]) -> LayoutNode:
    """Convert parsed tree into a LayoutNode tree with depth info.

    If the parsed tree has multiple roots (depth == 0), wrap them in a
    synthetic root so the layout is always a single tree.
    """
    max_depth = max((n.depth for n in nodes), default=0)

    def wrap(ax: AXNode) -> LayoutNode:
        ln = LayoutNode(
            ax=ax,
            bucket=classify_depth(ax, max_depth),
            depth=ax.depth,
            children=[wrap(c) for c in ax.children],
        )
        return ln

    roots = [n for n in nodes if n.depth == 0]
    if not roots:
        raise ValueError("Tree has no root-level nodes")
    if len(roots) == 1:
        return wrap(roots[0])

    synthetic = LayoutNode(
        ax=AXNode(role="AXGroup", depth=-1),
        bucket="root",
        depth=-1,
        children=[wrap(r) for r in roots],
    )
    # Shift depths up so synthetic root sits at 0.
    def shift(ln: LayoutNode, delta: int) -> None:
        ln.depth += delta
        for c in ln.children or []:
            shift(c, delta)
    shift(synthetic, 1)
    return synthetic


def assign_positions(root: LayoutNode, x_spacing: float, y_spacing: float) -> Tuple[float, float]:
    """Assign x/y in layout units. Leaves occupy x slots left-to-right;
    parents centre over their children.  Returns (width_in_units, depth)."""
    slot = [0]
    max_depth = [0]

    def walk(ln: LayoutNode) -> None:
        if ln.depth > max_depth[0]:
            max_depth[0] = ln.depth
        if not ln.children:
            ln.x = slot[0] * x_spacing
            slot[0] += 1
        else:
            for c in ln.children:
                walk(c)
            ln.x = (ln.children[0].x + ln.children[-1].x) / 2
        ln.y = ln.depth * y_spacing

    walk(root)
    return slot[0] * x_spacing, (max_depth[0] + 1) * y_spacing


def draw_tree(
    root: LayoutNode,
    canvas_w: int,
    canvas_h: int,
    title: Optional[str] = None,
    title_font: Optional[ImageFont.ImageFont] = None,
) -> Image.Image:
    """Render the layout tree onto a white canvas of given size."""
    img = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Collect nodes and compute bounding box of layout coords.
    all_nodes: List[LayoutNode] = []
    def collect(ln):
        all_nodes.append(ln)
        for c in ln.children or []:
            collect(c)
    collect(root)

    xs = [n.x for n in all_nodes]
    ys = [n.y for n in all_nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Margins — leave space above for the title.
    pad_x = 40
    pad_top = 60 if title else 40
    pad_bot = 40
    usable_w = canvas_w - 2 * pad_x
    usable_h = canvas_h - pad_top - pad_bot
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    sx = usable_w / span_x
    sy = usable_h / span_y

    def project(ln: LayoutNode) -> Tuple[int, int]:
        px = pad_x + int((ln.x - min_x) * sx)
        py = pad_top + int((ln.y - min_y) * sy)
        return px, py

    # Edges first (so circles overlay line endpoints).
    def draw_edges(ln: LayoutNode) -> None:
        px, py = project(ln)
        for c in ln.children or []:
            cx, cy = project(c)
            draw.line([(px, py), (cx, cy)], fill=EDGE_COLOR, width=2)
            draw_edges(c)
    draw_edges(root)

    # Nodes.
    for n in all_nodes:
        px, py = project(n)
        r = BUCKET_RADIUS[n.bucket]
        color = BUCKET_COLOR[n.bucket]
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=(0, 0, 0))

    # Title
    if title:
        font = title_font or _load_font(38)
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((canvas_w - tw) // 2, 20), title, fill=TEXT_COLOR, font=font)

    return img


# ---------------------------------------------------------------------------
# Legend


def draw_legend(width: int, height: int = 90, font: Optional[ImageFont.ImageFont] = None) -> Image.Image:
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font = font or _load_font(28)
    entries = [
        ("Root (window)",      COLOR_ROOT, BUCKET_RADIUS["root"]),
        ("Top container",      COLOR_TOP,  BUCKET_RADIUS["top"]),
        ("Mid group",          COLOR_MID,  BUCKET_RADIUS["mid"]),
        ("Leaf (UI element)",  COLOR_LEAF, BUCKET_RADIUS["leaf"]),
    ]

    # Measure widths
    gaps = 30
    segs = []
    for label, color, r in entries:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        seg_w = 2 * r + 10 + tw
        segs.append((label, color, r, tw, seg_w))
    total = sum(s[4] for s in segs) + gaps * (len(segs) - 1)
    x = (width - total) // 2
    cy = height // 2
    for label, color, r, tw, seg_w in segs:
        draw.ellipse([x, cy - r, x + 2 * r, cy + r], fill=color, outline=(0, 0, 0))
        draw.text((x + 2 * r + 8, cy - bbox[3] // 2 + 2), label, fill=TEXT_COLOR, font=font)
        x += seg_w + gaps
    return img


# ---------------------------------------------------------------------------
# Utilities


def _load_font(size: int) -> ImageFont.ImageFont:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def compose_figure(
    screenshot: Image.Image,
    tree_panels: List[Tuple[str, Image.Image]],
    legend: Image.Image,
    target_height: int = 900,
) -> Image.Image:
    """Stitch: [screenshot] [tree1] [tree2] ... with legend underneath."""
    # Rescale screenshot to target height.
    w0, h0 = screenshot.size
    new_h = target_height
    new_w = int(w0 * new_h / h0)
    screenshot_r = screenshot.resize((new_w, new_h), Image.LANCZOS)

    # Trees are already sized to target_height.
    tree_imgs = [t.resize((t.size[0] * new_h // t.size[1], new_h), Image.LANCZOS)
                 if t.size[1] != new_h else t
                 for _, t in tree_panels]

    total_w = screenshot_r.size[0] + sum(t.size[0] for t in tree_imgs) + 20 * (1 + len(tree_imgs))
    total_h = new_h + legend.size[1] + 40

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    x = 20
    canvas.paste(screenshot_r, (x, 20))
    x += screenshot_r.size[0] + 20
    for img in tree_imgs:
        canvas.paste(img, (x, 20))
        x += img.size[0] + 20

    # Legend centred at bottom.
    lx = (total_w - legend.size[0]) // 2
    canvas.paste(legend, (lx, new_h + 30))
    return canvas


# ---------------------------------------------------------------------------
# CLI


def parse_tree_arg(spec: str) -> Tuple[str, Path]:
    """Parse "label=path" or bare "path"."""
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        label, path = Path(spec).stem, spec
    return label, Path(path)


def render_one_tree_panel(
    tree_path: Path,
    label: str,
    panel_w: int,
    panel_h: int,
) -> Image.Image:
    text = tree_path.read_text(encoding="utf-8", errors="ignore")
    nodes = parse_tree(text)
    if not nodes:
        img = Image.new("RGB", (panel_w, panel_h), BG_COLOR)
        d = ImageDraw.Draw(img)
        d.text((20, 20), f"[{label}] empty tree", fill=(200, 40, 40), font=_load_font(18))
        return img
    root = build_layout(nodes)
    assign_positions(root, x_spacing=1.0, y_spacing=1.0)
    return draw_tree(root, panel_w, panel_h, title=label)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    image = Image.open(args.image).convert("RGB")

    # --- Screenshot overlay uses the first tree (or --overlay-tree). ---
    overlay_spec = args.overlay_tree or args.tree[0]
    overlay_label, overlay_path = parse_tree_arg(overlay_spec)
    overlay_nodes = parse_tree(overlay_path.read_text(encoding="utf-8", errors="ignore"))
    max_depth = max((n.depth for n in overlay_nodes), default=0)
    screenshot_panel = render_bboxes_on_image(
        image, overlay_nodes, max_depth, line_width=args.line_width
    )

    # --- One tree panel per --tree spec. ---
    panel_w = args.tree_panel_width
    panel_h = args.panel_height
    tree_panels: List[Tuple[str, Image.Image]] = []
    for spec in args.tree:
        label, path = parse_tree_arg(spec)
        tree_panels.append((label, render_one_tree_panel(path, label, panel_w, panel_h)))

    # --- Legend. ---
    # Width matches the final figure; rescale after compose by re-calling if needed.
    tentative_w = 2000
    legend = draw_legend(tentative_w, height=90)

    fig = compose_figure(
        screenshot_panel,
        tree_panels,
        legend,
        target_height=panel_h,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out_path)
    print(f"Saved: {out_path}  ({fig.size[0]}x{fig.size[1]})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a two-panel AX hierarchy figure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image", required=True,
                   help="Path to the screenshot.")
    p.add_argument("--tree", nargs="+", required=True,
                   help="One or more tree .txt paths. Use 'label=path' to set a panel title; "
                        "otherwise the file stem is used.")
    p.add_argument("--overlay-tree", default=None,
                   help="Which tree to overlay on the screenshot (defaults to first --tree).")
    p.add_argument("--out", required=True,
                   help="Output PNG path.")
    p.add_argument("--panel-height", type=int, default=900,
                   help="Height (px) of each panel.")
    p.add_argument("--tree-panel-width", type=int, default=900,
                   help="Width (px) per tree panel.")
    p.add_argument("--line-width", type=int, default=6,
                   help="Bbox outline thickness on the screenshot overlay.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
