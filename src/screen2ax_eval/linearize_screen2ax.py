"""
linearize_screen2ax.py

Converts the Screen2AX-Tree dataset from HuggingFace into a linearized,
token-efficient format suitable for VLM evaluation.

Differences from the original /workspace/linearize_screen2ax.py:
- Roles keep their original names (AXButton, not Button)
- Bounding boxes are [x1,y1,x2,y2] normalized to 0-1000 (not [x,y,w,h] pixels)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import tiktoken
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float_to_int(s: str) -> int:
    """Convert a string to int, returning 0 for non-finite floats."""
    v = float(s)
    if not (v == v) or v == float("inf") or v == float("-inf"):  # nan or inf
        return 0
    return round(v)


def _parse_position(position: Any) -> tuple[int, int]:
    """Parse a position string like '306.00;256.00' into (x, y)."""
    if not position:
        return 0, 0
    try:
        parts = str(position).replace(",", ";").split(";")
        return _safe_float_to_int(parts[0]), _safe_float_to_int(parts[1])
    except (ValueError, IndexError):
        return 0, 0


def _parse_size(size: Any) -> tuple[int, int]:
    """Parse a size string like '480;576' into (w, h)."""
    if not size:
        return 0, 0
    try:
        parts = str(size).replace(",", ";").split(";")
        return _safe_float_to_int(parts[0]), _safe_float_to_int(parts[1])
    except (ValueError, IndexError):
        return 0, 0


def _format_role(
    node: dict,
    role_mapping: Optional[dict[str, Optional[str]]] = None,
) -> Optional[str]:
    """Format role string, keeping the original AX prefix.

    Appends role_description in parentheses if it differs from the role name.
    If *role_mapping* is provided the role is remapped first; returns ``None``
    when the mapping drops the role.
    """
    raw_role = node.get("role", "") or ""
    if not raw_role:
        raw_role = "AXUnknown"

    if role_mapping is not None:
        mapped = role_mapping.get(raw_role)
        if mapped is None:
            return None
        raw_role = mapped

    role_desc = node.get("role_description") or ""
    # Strip AX prefix just for comparison with role_description
    bare = raw_role[2:] if raw_role.startswith("AX") else raw_role
    if role_desc and role_desc.lower() != bare.lower():
        return f"{raw_role}({role_desc})"
    return raw_role


def _tree_depth(node: dict) -> int:
    """Return the maximum depth of a tree rooted at node."""
    children = node.get("children") or []
    if not children:
        return 1
    return 1 + max(_tree_depth(c) for c in children)


def _node_count(node: dict) -> int:
    """Return the total number of nodes in the tree."""
    children = node.get("children") or []
    return 1 + sum(_node_count(c) for c in children)


def _collect_roles(node: dict, counter: Counter) -> None:
    """Recursively collect all role values into counter."""
    role = node.get("role", "")
    if role:
        counter[role] += 1
    for child in node.get("children") or []:
        _collect_roles(child, counter)


# ---------------------------------------------------------------------------
# Coordinate normalization
# ---------------------------------------------------------------------------

def _normalize_bbox(
    x: int, y: int, w: int, h: int,
    img_width: int, img_height: int,
    target_range: int = 1000,
    scale_factor: float = 1.0,
) -> tuple[int, int, int, int]:
    """Convert pixel [x, y, w, h] to normalized [x1, y1, x2, y2] in 0-target_range.

    If *scale_factor* != 1 the normalised coordinates are multiplied by it
    (useful when GT boxes need rescaling to match a different resolution).
    """
    if img_width <= 0 or img_height <= 0:
        return 0, 0, 0, 0

    x1 = int(x / img_width * target_range * scale_factor)
    y1 = int(y / img_height * target_range * scale_factor)
    x2 = int((x + w) / img_width * target_range * scale_factor)
    y2 = int((y + h) / img_height * target_range * scale_factor)
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Tree pruning (unchanged from original)
# ---------------------------------------------------------------------------

def prune_tree(tree_dict: dict) -> dict:
    """Prune the accessibility tree before linearization.

    Applies the following transformations:
    - Remove nodes where ``visibility`` or ``visible-to-user`` is False.
    - Collapse AXGroup nodes with no name/description/value and exactly one
      child by replacing the group with its child.
    - Remove empty leaf AXGroup nodes (no children, no meaningful attributes).
    - Deduplicate consecutive identical siblings (same role/name/value/desc),
      keeping the first and appending a ``(xN)`` marker.
    """
    if tree_dict.get("visibility") is False or tree_dict.get("visible-to-user") is False:
        return {}

    role = tree_dict.get("role", "")
    name = tree_dict.get("name") or ""
    description = tree_dict.get("description") or ""
    value = tree_dict.get("value")
    has_meaningful = bool(name.strip() or description.strip() or value is not None)

    raw_children: list[dict] = tree_dict.get("children") or []
    pruned_children: list[dict] = []
    for child in raw_children:
        result = prune_tree(child)
        if result:
            pruned_children.append(result)

    if role == "AXGroup" and not pruned_children and not has_meaningful:
        return {}

    if role == "AXGroup" and len(pruned_children) == 1 and not has_meaningful:
        return pruned_children[0]

    deduped: list[dict] = []
    i = 0
    while i < len(pruned_children):
        node = pruned_children[i]
        count = 1
        while (
            i + count < len(pruned_children)
            and pruned_children[i + count].get("role") == node.get("role")
            and pruned_children[i + count].get("name") == node.get("name")
            and pruned_children[i + count].get("value") == node.get("value")
            and pruned_children[i + count].get("description") == node.get("description")
        ):
            count += 1
        if count > 1:
            marked = dict(node)
            marker = f"(\u00d7{count})"
            existing = marked.get("name") or ""
            marked["name"] = f"{existing} {marker}".strip() if existing else marker
            deduped.append(marked)
        else:
            deduped.append(node)
        i += count

    result = dict(tree_dict)
    result["children"] = deduped
    return result


# ---------------------------------------------------------------------------
# Post-mapping collapse (for simplified roles)
# ---------------------------------------------------------------------------

def _node_bbox_xyxy(node: dict) -> tuple[int, int, int, int]:
    """Return the node's pixel [x1, y1, x2, y2] from position/size fields."""
    x, y = _parse_position(node.get("position"))
    w, h = _parse_size(node.get("size"))
    return x, y, x + w, y + h


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU between two xyxy boxes. Returns 0 for zero-area boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _has_meaningful_attrs(node: dict) -> bool:
    """True if the node has a name, description, or value worth preserving."""
    name = (node.get("name") or "").strip()
    desc = (node.get("description") or "").strip()
    value = node.get("value")
    return bool(name or desc or value is not None)


def _map_and_collapse(
    tree_dict: dict,
    role_mapping: dict[str, Optional[str]],
    iou_threshold: float = 0.95,
) -> Optional[dict]:
    """Apply *role_mapping* to a tree dict and collapse redundant same-role chains.

    Transformations:
    - If a node's role maps to ``None`` (or is absent from the mapping), the
      node is dropped and its (recursively mapped) children are returned to be
      spliced into the parent's child list.
    - If a node ends up with exactly one mapped child, both share the same
      (mapped) role, their bboxes have IoU >= ``iou_threshold``, and the
      parent has no meaningful attributes, the parent is replaced by the child.

    Returns a new tree dict, or ``None`` if the node and all descendants were
    dropped. Callers should splice in lists returned via the children slot.
    """
    # Recursively process children first; each may expand (if a child was
    # dropped we get its grandchildren instead) or contract to None.
    new_children: list[dict] = []
    for child in tree_dict.get("children") or []:
        mapped = _map_and_collapse(child, role_mapping, iou_threshold)
        if mapped is None:
            continue
        if isinstance(mapped, list):
            new_children.extend(mapped)
        else:
            new_children.append(mapped)

    raw_role = tree_dict.get("role", "") or "AXUnknown"
    mapped_role = role_mapping.get(raw_role)

    if mapped_role is None:
        # Drop this node; promote its children to the parent's level.
        # Signal by returning a list via a sentinel wrapper: we return the
        # first child if any, and append the rest by using a plain list.
        if not new_children:
            return None
        if len(new_children) == 1:
            return new_children[0]
        return new_children  # type: ignore[return-value]

    # Build a copy of this node with mapped role and processed children
    result = dict(tree_dict)
    result["role"] = mapped_role
    result["children"] = new_children

    # Collapse: parent has no meaningful attrs, exactly one child, same
    # mapped role, and bboxes near-identical -> replace parent with child.
    if (
        len(new_children) == 1
        and not _has_meaningful_attrs(result)
        and new_children[0].get("role") == mapped_role
        and _bbox_iou(_node_bbox_xyxy(result), _node_bbox_xyxy(new_children[0])) >= iou_threshold
    ):
        return new_children[0]

    return result


# ---------------------------------------------------------------------------
# Linearization
# ---------------------------------------------------------------------------

def linearize_tree(
    tree_dict: dict,
    img_width: int,
    img_height: int,
    max_depth: Optional[int] = None,
    normalize_range: int = 1000,
    scale_factor: float = 1.0,
    role_mapping: Optional[dict[str, Optional[str]]] = None,
    _current_depth: int = 0,
    _indent: str = "",
) -> str:
    """Recursively convert a nested AX tree dict to an indented text format.

    Each node is formatted as::

        AXRole(role_description) [x1,y1,x2,y2] name="..." value="..." desc="..."

    where coordinates are normalized to [0, normalize_range].
    If *role_mapping* is provided, the whole tree is first remapped and
    redundant same-role chains collapsed via :func:`_map_and_collapse`.
    """
    if not tree_dict:
        return ""

    # At the top of the recursion, apply role mapping + same-role collapse
    # once to the whole tree. Recursive calls see an already-mapped tree
    # and pass ``role_mapping=None`` to avoid re-mapping.
    if role_mapping is not None and _current_depth == 0:
        mapped = _map_and_collapse(tree_dict, role_mapping)
        if mapped is None:
            return ""
        if isinstance(mapped, list):
            lines: list[str] = []
            for node in mapped:
                child_text = linearize_tree(
                    node,
                    img_width=img_width,
                    img_height=img_height,
                    max_depth=max_depth,
                    normalize_range=normalize_range,
                    scale_factor=scale_factor,
                    role_mapping=None,
                    _current_depth=0,
                    _indent=_indent,
                )
                if child_text:
                    lines.append(child_text)
            return "\n".join(lines)
        tree_dict = mapped

    role_label = _format_role(tree_dict, role_mapping=None)
    if role_label is None:
        return ""

    # Bounding box: parse pixel coords, convert to normalized xyxy
    x, y = _parse_position(tree_dict.get("position"))
    w, h = _parse_size(tree_dict.get("size"))
    x1, y1, x2, y2 = _normalize_bbox(x, y, w, h, img_width, img_height, normalize_range, scale_factor)
    bbox = f"[{x1},{y1},{x2},{y2}]"

    # Attributes
    attrs: list[str] = []
    name = tree_dict.get("name") or ""
    if name.strip():
        attrs.append(f'name="{name.strip()}"')

    value = tree_dict.get("value")
    if value is not None:
        # Collapse newlines to spaces so multiline values don't break the
        # one-line-per-node linearized format.
        value_str = re.sub(r"\s*\n\s*", " ", str(value)).strip()
        attrs.append(f'value="{value_str}"')

    desc = tree_dict.get("description") or ""
    if desc.strip():
        attrs.append(f'desc="{desc.strip()}"')

    attr_str = (" " + " ".join(attrs)) if attrs else ""
    line = f"{_indent}{role_label} {bbox}{attr_str}"

    lines = [line]

    # Children
    children: list[dict] = tree_dict.get("children") or []
    if children:
        if max_depth is not None and _current_depth >= max_depth - 1:
            lines.append(f"{_indent}  ...")
        else:
            for child in children:
                child_text = linearize_tree(
                    child,
                    img_width=img_width,
                    img_height=img_height,
                    max_depth=max_depth,
                    normalize_range=normalize_range,
                    scale_factor=scale_factor,
                    role_mapping=None,
                    _current_depth=_current_depth + 1,
                    _indent=_indent + "  ",
                )
                if child_text:
                    lines.append(child_text)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(dataset) -> dict:
    """Compute statistics over the full dataset."""
    enc = tiktoken.get_encoding("cl100k_base")

    depths: list[int] = []
    node_counts: list[int] = []
    text_lengths: list[int] = []
    token_counts: list[int] = []
    role_counter: Counter = Counter()
    length_hist: defaultdict[int, int] = defaultdict(int)

    for sample in tqdm(dataset, desc="Computing stats"):
        try:
            tree = json.loads(sample["accessibility"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

        tree = prune_tree(tree)
        if not tree:
            continue

        image = sample["image"]
        img_w, img_h = image.size

        depths.append(_tree_depth(tree))
        node_counts.append(_node_count(tree))

        text = linearize_tree(tree, img_width=img_w, img_height=img_h)
        tlen = len(text)
        text_lengths.append(tlen)
        token_counts.append(len(enc.encode(text)))

        _collect_roles(tree, role_counter)

        bucket = (tlen // 500) * 500
        length_hist[bucket] += 1

    def _agg(lst: list[int]) -> dict:
        if not lst:
            return {"avg": 0, "min": 0, "max": 0}
        return {
            "avg": round(sum(lst) / len(lst), 1),
            "min": min(lst),
            "max": max(lst),
        }

    def _percentiles(lst: list[int], pcts: list[int]) -> dict[str, int]:
        if not lst:
            return {f"p{p}": 0 for p in pcts}
        s = sorted(lst)
        n = len(s)
        result: dict[str, int] = {}
        for p in pcts:
            idx = max(0, min(n - 1, round(p / 100 * n) - 1 if p < 100 else n - 1))
            result[f"p{p}"] = s[idx]
        return result

    depth_pcts = _percentiles(depths, [5, 25, 50, 75, 90, 95, 99])

    return {
        "total_samples": len(depths),
        "depth": {**_agg(depths), **depth_pcts},
        "node_count": _agg(node_counts),
        "text_length": _agg(text_lengths),
        "token_count": _agg(token_counts),
        "top_roles": role_counter.most_common(20),
        "length_histogram": dict(sorted(length_hist.items())),
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_lava_format(
    dataset,
    output_dir: str | Path,
    max_depth: Optional[int] = None,
    train_ratio: float = 0.85,
    scale_factor: float = 1.0,
    role_mapping: Optional[dict[str, Optional[str]]] = None,
) -> None:
    """Export dataset to LLaVA-format JSON for VLM fine-tuning.

    Also writes individual annotation .txt files to {output_dir}/annotations/.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    HUMAN_PROMPT = (
        "<image>\n"
        "Generate the complete accessibility tree for this UI screenshot. "
        "Output each element on a new line with indentation showing hierarchy. "
        "Format: AXRole(subrole) [x1,y1,x2,y2] attributes"
    )

    entries: list[dict] = []

    for idx, sample in enumerate(tqdm(dataset, desc="Exporting")):
        try:
            tree = json.loads(sample["accessibility"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

        tree = prune_tree(tree)
        if not tree:
            continue

        image = sample["image"]
        img_w, img_h = image.size

        linearized = linearize_tree(tree, img_width=img_w, img_height=img_h, max_depth=max_depth, scale_factor=scale_factor, role_mapping=role_mapping)
        if not linearized.strip():
            continue

        # Strip unpaired surrogates that are invalid in UTF-8
        linearized = linearized.encode("utf-8", errors="ignore").decode("utf-8")

        # Save image
        img_filename = f"{idx}.png"
        image.save(images_dir / img_filename)

        # Save annotation
        ann_filename = f"{idx}.txt"
        with open(annotations_dir / ann_filename, "w", encoding="utf-8") as f:
            f.write(linearized)

        entries.append(
            {
                "id": f"screen2ax_{idx}",
                "image": f"images/{img_filename}",
                "conversations": [
                    {"from": "human", "value": HUMAN_PROMPT},
                    {"from": "gpt", "value": linearized},
                ],
            }
        )

    # Split
    split_idx = round(len(entries) * train_ratio)
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]

    def _clean(text: str) -> str:
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def _clean_entry(entry: dict) -> dict:
        for conv in entry["conversations"]:
            conv["value"] = _clean(conv["value"])
        return entry

    train_entries = [_clean_entry(e) for e in train_entries]
    val_entries = [_clean_entry(e) for e in val_entries]

    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_entries, f, ensure_ascii=False, indent=2)

    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_entries, f, ensure_ascii=False, indent=2)

    config = {
        "dataset": "macpaw-research/Screen2AX-Tree",
        "max_depth": max_depth,
        "train_ratio": train_ratio,
        "total_samples": len(entries),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "scale_factor": scale_factor,
        "bbox_format": "xyxy_normalized_0_1000",
        "role_format": "original_with_AX_prefix",
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(
        f"\nExport complete: {len(train_entries)} train / {len(val_entries)} val samples"
        f"\nOutput: {output_dir}"
        f"\nAnnotations: {annotations_dir}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_dataset_split(split: str = "train"):
    print(f"Loading macpaw-research/Screen2AX-Tree ({split})\u2026")
    return load_dataset("macpaw-research/Screen2AX-Tree", split=split)


def cmd_stats(args: argparse.Namespace) -> None:
    dataset = _load_dataset_split()
    stats = compute_stats(dataset)

    print("\n=== Screen2AX-Tree Dataset Statistics ===\n")
    print(f"Total samples : {stats['total_samples']}")
    d = stats['depth']
    print(f"Tree depth    : avg={d['avg']}  min={d['min']}  max={d['max']}")
    print(f"  percentiles : p5={d['p5']}  p25={d['p25']}  p50={d['p50']}  p75={d['p75']}  p90={d['p90']}  p95={d['p95']}  p99={d['p99']}")
    print(f"Node count    : avg={stats['node_count']['avg']}  min={stats['node_count']['min']}  max={stats['node_count']['max']}")
    print(f"Text length   : avg={stats['text_length']['avg']}  min={stats['text_length']['min']}  max={stats['text_length']['max']}")
    print(f"Token count   : avg={stats['token_count']['avg']}  min={stats['token_count']['min']}  max={stats['token_count']['max']}")

    print("\nTop 20 AX Roles:")
    for role, cnt in stats["top_roles"]:
        print(f"  {role:<30} {cnt}")

    print("\nLinearized Text Length Histogram (bucket=500 chars):")
    for bucket, cnt in stats["length_histogram"].items():
        bar = "#" * min(cnt, 60)
        print(f"  {bucket:>6} - {bucket+499:<6}  {bar} ({cnt})")


def cmd_preview(args: argparse.Namespace) -> None:
    dataset = _load_dataset_split()
    max_depth: Optional[int] = args.max_depth

    shown = 0
    for sample in dataset:
        if shown >= args.n:
            break
        try:
            tree = json.loads(sample["accessibility"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        tree = prune_tree(tree)
        if not tree:
            continue

        image = sample["image"]
        img_w, img_h = image.size

        role_mapping = config.SIMPLIFIED_ROLE_MAPPING if args.simplified_roles else None
        text = linearize_tree(tree, img_width=img_w, img_height=img_h, max_depth=max_depth, scale_factor=args.scale_factor, role_mapping=role_mapping)
        print(f"\n{'='*60}")
        print(f"Sample {shown + 1}  (depth={_tree_depth(tree)}, nodes={_node_count(tree)}, img={img_w}x{img_h})")
        print("="*60)
        print(text)
        shown += 1


def cmd_export(args: argparse.Namespace) -> None:
    max_depth: Optional[int] = args.max_depth
    role_mapping = config.SIMPLIFIED_ROLE_MAPPING if args.simplified_roles else None
    dataset = _load_dataset_split()
    export_lava_format(
        dataset,
        output_dir=args.output_dir,
        max_depth=max_depth,
        train_ratio=args.train_ratio,
        scale_factor=args.scale_factor,
        role_mapping=role_mapping,
    )


def cmd_export_trees(args: argparse.Namespace) -> None:
    """Save the raw (pruned) tree dict as individual JSON files."""
    dataset = _load_dataset_split()
    output_dir = Path(args.output_dir)
    trees_dir = output_dir / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx, sample in enumerate(tqdm(dataset, desc="Exporting trees")):
        try:
            tree = json.loads(sample["accessibility"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

        if args.prune:
            tree = prune_tree(tree)
        if not tree:
            continue

        # Strip unpaired surrogates that are invalid in UTF-8
        tree_json = json.dumps(tree, ensure_ascii=False, indent=2)
        tree_json = tree_json.encode("utf-8", errors="ignore").decode("utf-8")
        with open(trees_dir / f"{idx}.json", "w", encoding="utf-8") as f:
            f.write(tree_json)
        saved += 1

    print(f"\nSaved {saved} tree JSON files to {trees_dir}")


def cmd_tokencount(args: argparse.Namespace) -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    dataset = _load_dataset_split()

    depths: list[Optional[int]] = []
    for d in args.max_depth:
        if d.lower() == "none":
            depths.append(None)
        else:
            depths.append(int(d))

    # Collect trees and image dimensions once
    samples: list[tuple[dict, int, int]] = []
    for sample in tqdm(dataset, desc="Parsing trees"):
        try:
            tree = json.loads(sample["accessibility"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        tree = prune_tree(tree)
        if tree:
            image = sample["image"]
            samples.append((tree, image.size[0], image.size[1]))

    print(f"\nComparing token counts across {len(samples)} samples\n")
    print(f"{'Depth':<10} {'Avg tokens':>12} {'Min':>8} {'Max':>8} {'Total':>12}")
    print("-" * 54)

    role_mapping = config.SIMPLIFIED_ROLE_MAPPING if args.simplified_roles else None
    for depth in depths:
        counts = []
        for tree, img_w, img_h in tqdm(samples, desc=f"depth={depth}", leave=False):
            text = linearize_tree(tree, img_width=img_w, img_height=img_h, max_depth=depth, scale_factor=args.scale_factor, role_mapping=role_mapping)
            counts.append(len(enc.encode(text)))
        avg = sum(counts) / len(counts) if counts else 0
        label = str(depth) if depth is not None else "None"
        print(
            f"{label:<10} {avg:>12.1f} {min(counts):>8} {max(counts):>8} {sum(counts):>12}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen2AX-Tree linearization (normalized xyxy + AX roles)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # stats
    sub.add_parser("stats", help="Print dataset statistics")

    # preview
    p_preview = sub.add_parser("preview", help="Show N linearized examples")
    p_preview.add_argument("--n", type=int, default=5, help="Number of examples to show")
    p_preview.add_argument(
        "--max-depth", type=lambda x: None if x.lower() == "none" else int(x),
        default=None, metavar="DEPTH",
        help="Max tree depth (integer or 'None')"
    )
    p_preview.add_argument("--scale-factor", type=float, default=1.0,
                           help="Multiply GT coordinates by this factor (default: 1)")
    p_preview.add_argument("--simplified-roles", action="store_true",
                           help="Use 7-class simplified role mapping from Screen2AX paper")

    # export
    p_export = sub.add_parser("export", help="Export dataset in LLaVA format")
    p_export.add_argument("--output-dir", default="./data_llava", help="Output directory")
    p_export.add_argument(
        "--max-depth", type=lambda x: None if x.lower() == "none" else int(x),
        default=None, metavar="DEPTH",
        help="Max tree depth (integer or 'None')"
    )
    p_export.add_argument("--train-ratio", type=float, default=0.85)
    p_export.add_argument("--scale-factor", type=float, default=1.0,
                          help="Multiply GT coordinates by this factor (default: 1)")
    p_export.add_argument("--simplified-roles", action="store_true",
                          help="Use 7-class simplified role mapping from Screen2AX paper")

    # export-trees
    p_trees = sub.add_parser("export-trees", help="Save raw tree dicts as individual JSON files")
    p_trees.add_argument("--output-dir", default="./data_llava", help="Output directory (trees saved to <output-dir>/trees/)")
    p_trees.add_argument("--no-prune", dest="prune", action="store_false",
                         help="Skip pruning — save the original tree as-is")

    # tokencount
    p_tok = sub.add_parser("tokencount", help="Compare token counts at different depths")
    p_tok.add_argument(
        "--max-depth", nargs="+", default=["3", "4", "5", "6", "None"],
        metavar="DEPTH",
        help="Space-separated depth values (integers or 'None')"
    )
    p_tok.add_argument("--scale-factor", type=float, default=1.0,
                       help="Multiply GT coordinates by this factor (default: 1)")
    p_tok.add_argument("--simplified-roles", action="store_true",
                       help="Use 7-class simplified role mapping from Screen2AX paper")

    args = parser.parse_args()

    dispatch = {
        "stats": cmd_stats,
        "preview": cmd_preview,
        "export": cmd_export,
        "export-trees": cmd_export_trees,
        "tokencount": cmd_tokencount,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
