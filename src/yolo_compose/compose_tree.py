"""Compose YOLO two-stage detections into a linearized AX tree.

Reads the per-image detection JSONs written by infer_yolo_two_stage.py and
emits {stem}.txt files in the same format as VLM predictions, ready for
screen2ax_eval/evaluate.py.

Algorithm:
  1. Prepend a synthetic root AXGroup [0,0,1000,1000] so every leaf has a
     container even if Stage 2 found nothing.
  2. Assign each group's parent = smallest other group that contains it
     (90% overlap by default). Default to root if none found.
  3. Assign each leaf's parent the same way.
  4. Sort siblings in reading order (y1 bucketed, then x1).
  5. DFS-walk and emit "  " * depth + "Role [x1,y1,x2,y2]".

Containment uses soft overlap (intersection / child_area) not strict
geometric containment, because YOLO boxes don't perfectly align.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("yolo_compose.compose")

Bbox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) in 0-1000


# ---------------------------------------------------------------------------
# Geometry


def area(b: Bbox) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersection_area(a: Bbox, b: Bbox) -> int:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def contains(parent: Bbox, child: Bbox, overlap_threshold: float) -> bool:
    ca = area(child)
    if ca <= 0:
        return False
    return intersection_area(parent, child) / ca >= overlap_threshold


# ---------------------------------------------------------------------------
# Tree


@dataclasses.dataclass
class Node:
    role: str
    bbox: Bbox
    children: List["Node"] = dataclasses.field(default_factory=list)
    # For stable tie-breaking only
    score: float = 1.0
    is_root: bool = False


def smallest_containing(
    candidates: List[Node],
    child_bbox: Bbox,
    exclude: Optional[Node],
    overlap_threshold: float,
) -> Optional[Node]:
    """Return the smallest-area candidate that contains child_bbox."""
    best: Optional[Node] = None
    best_area = None
    for c in candidates:
        if c is exclude:
            continue
        if not contains(c.bbox, child_bbox, overlap_threshold):
            continue
        a = area(c.bbox)
        if best_area is None or a < best_area or (
            a == best_area and (c.bbox[1], c.bbox[0]) < (best.bbox[1], best.bbox[0])
        ):
            best = c
            best_area = a
    return best


def reading_order_sort(nodes: List[Node], y_bucket: int = 5) -> List[Node]:
    """Sort siblings top-to-bottom, left-to-right.

    y is bucketed so two elements on roughly the same row stay ordered by x.
    """
    def key(n: Node):
        return (n.bbox[1] // y_bucket, n.bbox[0], n.bbox[2], n.bbox[3], n.role)
    return sorted(nodes, key=key)


# ---------------------------------------------------------------------------
# Composition


def compose(
    leaves: List[Dict],
    groups: List[Dict],
    overlap_threshold: float = 0.9,
    dedupe_iou: float = 0.95,
) -> Node:
    """Build a nested AXNode tree from detection lists.

    leaves: list of {"role": str, "bbox": [x1,y1,x2,y2], "score": float}
    groups: list of {"bbox": [x1,y1,x2,y2], "score": float}
    """
    # Dedupe near-identical boxes within each list (belt-and-suspenders
    # against YOLO NMS leaking duplicates).
    leaves = _dedupe(leaves, with_role=True, iou=dedupe_iou)
    groups = _dedupe(groups, with_role=False, iou=dedupe_iou)

    root = Node(role="AXGroup", bbox=(0, 0, 1000, 1000), is_root=True)

    group_nodes: List[Node] = [
        Node(role="AXGroup", bbox=tuple(g["bbox"]), score=float(g.get("score", 1.0)))
        for g in groups
    ]
    # Drop groups that exactly cover the whole canvas (they'd shadow root).
    group_nodes = [g for g in group_nodes if not (
        g.bbox[0] <= 0 and g.bbox[1] <= 0 and g.bbox[2] >= 1000 and g.bbox[3] >= 1000
    )]

    containers = [root] + group_nodes

    # Groups first — nest smaller groups inside bigger ones.
    # Process largest → smallest so when we look up parents we've only
    # considered candidates at least as large as the current group.
    for g in sorted(group_nodes, key=lambda n: area(n.bbox), reverse=True):
        parent = smallest_containing(containers, g.bbox, exclude=g,
                                     overlap_threshold=overlap_threshold)
        if parent is None:
            parent = root
        parent.children.append(g)

    # Then leaves.
    for leaf_info in leaves:
        leaf = Node(
            role=leaf_info["role"],
            bbox=tuple(leaf_info["bbox"]),
            score=float(leaf_info.get("score", 1.0)),
        )
        parent = smallest_containing(containers, leaf.bbox, exclude=None,
                                     overlap_threshold=overlap_threshold)
        if parent is None:
            parent = root
        parent.children.append(leaf)

    _sort_subtree(root)
    return root


def _sort_subtree(node: Node) -> None:
    node.children = reading_order_sort(node.children)
    for c in node.children:
        _sort_subtree(c)


def _iou(a: Bbox, b: Bbox) -> float:
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    ua = area(a) + area(b) - inter
    return inter / ua if ua > 0 else 0.0


def _dedupe(items: List[Dict], with_role: bool, iou: float) -> List[Dict]:
    """Remove boxes with IoU > iou to an earlier (higher-score) box."""
    items = sorted(items, key=lambda d: -float(d.get("score", 0.0)))
    kept: List[Dict] = []
    for it in items:
        bb = tuple(it["bbox"])
        role = it.get("role") if with_role else None
        dup = False
        for k in kept:
            if with_role and k.get("role") != role:
                continue
            if _iou(bb, tuple(k["bbox"])) > iou:
                dup = True
                break
        if not dup:
            kept.append(it)
    return kept


# ---------------------------------------------------------------------------
# Linearization


def linearize(root: Node, drop_root: bool = True) -> str:
    lines: List[str] = []

    def walk(node: Node, depth: int) -> None:
        bbox_str = ",".join(str(int(v)) for v in node.bbox)
        lines.append(f"{'  ' * depth}{node.role} [{bbox_str}]")
        for c in node.children:
            walk(c, depth + 1)

    if drop_root:
        if not root.children:
            # No detections at all — return an empty tree.
            return ""
        # Emit root's children at depth 0, skipping the synthetic root.
        # If the synthetic root is the only container and its children are
        # shallow, we lose no information.
        for c in root.children:
            walk(c, 0)
    else:
        walk(root, 0)

    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# CLI driver


def compose_split(
    detections_dir: Path,
    output_dir: Path,
    overlap_threshold: float,
    dedupe_iou: float,
    keep_root: bool,
) -> Dict:
    parsed_dir = output_dir / "parsed"
    meta_dir = output_dir / "metadata"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(detections_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No detection JSONs in {detections_dir}")

    n = 0
    total_leaves = 0
    total_groups = 0
    total_depth = 0
    empty = 0

    for det_path in files:
        data = json.loads(det_path.read_text())
        stem = data.get("stem", det_path.stem)

        tree = compose(
            data.get("leaves", []),
            data.get("groups", []),
            overlap_threshold=overlap_threshold,
            dedupe_iou=dedupe_iou,
        )
        text = linearize(tree, drop_root=not keep_root)

        (parsed_dir / f"{stem}.txt").write_text(text, encoding="utf-8")

        depth = _max_depth(tree) - (0 if keep_root else 1)
        total_depth += max(0, depth)

        leaves_count = sum(1 for _ in _iter_nodes(tree) if _ is not tree and not _.children and _.role != "AXGroup")
        groups_count = sum(1 for _ in _iter_nodes(tree) if _ is not tree and _.role == "AXGroup")

        (meta_dir / f"{stem}.json").write_text(json.dumps({
            "stem": stem,
            "leaves": leaves_count,
            "groups": groups_count,
            "max_depth": max(0, depth),
            "overlap_threshold": overlap_threshold,
            "dedupe_iou": dedupe_iou,
        }, ensure_ascii=False))

        if not text.strip():
            empty += 1
        n += 1
        total_leaves += leaves_count
        total_groups += groups_count

    stats = {
        "samples": n,
        "empty_trees": empty,
        "avg_leaves": total_leaves / n if n else 0.0,
        "avg_groups": total_groups / n if n else 0.0,
        "avg_depth":  total_depth / n if n else 0.0,
        "overlap_threshold": overlap_threshold,
        "dedupe_iou": dedupe_iou,
    }
    return stats


def _iter_nodes(node: Node):
    yield node
    for c in node.children:
        yield from _iter_nodes(c)


def _max_depth(node: Node, d: int = 0) -> int:
    if not node.children:
        return d
    return max(_max_depth(c, d + 1) for c in node.children)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    stats = compose_split(
        detections_dir=Path(args.detections_dir),
        output_dir=Path(args.output_dir),
        overlap_threshold=args.overlap_threshold,
        dedupe_iou=args.dedupe_iou,
        keep_root=args.keep_root,
    )
    print("=== compose ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<20} {v:.2f}")
        else:
            print(f"  {k:<20} {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--detections-dir", required=True,
                   help="Dir with per-image JSONs from infer_yolo_two_stage.py")
    p.add_argument("--output-dir", required=True,
                   help="Root output dir (will create parsed/ and metadata/)")
    p.add_argument("--overlap-threshold", type=float, default=0.9,
                   help="Child is 'inside' parent when intersection/child_area >= this")
    p.add_argument("--dedupe-iou", type=float, default=0.95,
                   help="Collapse boxes of the same class with IoU > this")
    p.add_argument("--keep-root", action="store_true",
                   help="Emit the synthetic root AXGroup as the outer line")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
