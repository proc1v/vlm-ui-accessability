"""
linearize_screen2ax.py

Converts the Screen2AX-Tree dataset from HuggingFace into a linearized,
token-efficient format suitable for VLM fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import tiktoken
from datasets import load_dataset
from tqdm import tqdm


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


def _strip_ax(role: str) -> str:
    """Strip 'AX' prefix from role string, e.g. 'AXButton' -> 'Button'."""
    if role and role.startswith("AX"):
        return role[2:]
    return role or "Unknown"


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
# 2. Tree pruning
# ---------------------------------------------------------------------------

def prune_tree(tree_dict: dict) -> dict:
    """Prune the accessibility tree before linearization.

    Applies the following transformations:
    - Remove nodes where ``visibility`` or ``visible-to-user`` is False.
    - Collapse AXGroup nodes with no name/description/value and exactly one
      child by replacing the group with its child.
    - Remove empty leaf AXGroup nodes (no children, no meaningful attributes).
    - Deduplicate consecutive identical siblings (same role/name/value/desc),
      keeping the first and appending a ``(×N)`` marker.

    Parameters
    ----------
    tree_dict:
        A single AX tree node (dict with optional ``children`` list).

    Returns
    -------
    dict
        The pruned tree, or an empty dict if the node should be removed.
    """
    # Visibility check
    if tree_dict.get("visibility") is False or tree_dict.get("visible-to-user") is False:
        return {}

    role = tree_dict.get("role", "")
    name = tree_dict.get("name") or ""
    description = tree_dict.get("description") or ""
    value = tree_dict.get("value")
    has_meaningful = bool(name.strip() or description.strip() or value is not None)

    # Recurse into children first
    raw_children: list[dict] = tree_dict.get("children") or []
    pruned_children: list[dict] = []
    for child in raw_children:
        result = prune_tree(child)
        if result:
            pruned_children.append(result)

    # Remove empty leaf AXGroup
    if role == "AXGroup" and not pruned_children and not has_meaningful:
        return {}

    # Collapse AXGroup chain (single child, no meaningful attrs)
    if role == "AXGroup" and len(pruned_children) == 1 and not has_meaningful:
        return pruned_children[0]

    # Deduplicate consecutive identical siblings
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
            marker = f"(×{count})"
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
# 1. Linearization
# ---------------------------------------------------------------------------

def linearize_tree(
    tree_dict: dict,
    max_depth: Optional[int] = None,
    _current_depth: int = 0,
    _indent: str = "",
) -> str:
    """Recursively convert a nested AX tree dict to an indented text format.

    Each node is formatted as::

        ROLE[(role_description)] [x,y,w,h] name="..." value="..." desc="..."

    Parameters
    ----------
    tree_dict:
        A single AX tree node.
    max_depth:
        Maximum recursion depth. ``None`` means unlimited.
    _current_depth:
        Internal: current recursion depth (used by recursive calls).
    _indent:
        Internal: current indentation string (used by recursive calls).

    Returns
    -------
    str
        Linearized text representation of the tree.
    """
    if not tree_dict:
        return ""

    # Role
    raw_role = tree_dict.get("role", "") or ""
    role_label = _strip_ax(raw_role)

    role_desc = tree_dict.get("role_description") or ""
    # Only append role_description if it adds info beyond the bare role name
    if role_desc and role_desc.lower() != role_label.lower():
        role_label = f"{role_label}({role_desc})"

    # Bounding box
    x, y = _parse_position(tree_dict.get("position"))
    w, h = _parse_size(tree_dict.get("size"))
    bbox = f"[{x},{y},{w},{h}]"

    # Attributes
    attrs: list[str] = []
    name = tree_dict.get("name") or ""
    if name.strip():
        attrs.append(f'name="{name.strip()}"')

    value = tree_dict.get("value")
    if value is not None:
        attrs.append(f'value="{value}"')

    desc = tree_dict.get("description") or ""
    if desc.strip():
        attrs.append(f'desc="{desc.strip()}"')

    attr_str = (" " + " ".join(attrs)) if attrs else ""
    line = f"{_indent}{role_label} {bbox}{attr_str}"

    lines: list[str] = [line]

    # Children
    children: list[dict] = tree_dict.get("children") or []
    if children:
        if max_depth is not None and _current_depth >= max_depth - 1:
            lines.append(f"{_indent}  ...")
        else:
            for child in children:
                child_text = linearize_tree(
                    child,
                    max_depth=max_depth,
                    _current_depth=_current_depth + 1,
                    _indent=_indent + "  ",
                )
                if child_text:
                    lines.append(child_text)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Statistics
# ---------------------------------------------------------------------------

def compute_stats(dataset) -> dict:
    """Compute statistics over the full dataset.

    Parameters
    ----------
    dataset:
        A HuggingFace ``Dataset`` object with ``image`` and ``accessibility``
        columns.

    Returns
    -------
    dict
        Dictionary with keys: total_samples, depth_*, node_count_*,
        text_length_*, token_count_*, top_roles, length_histogram.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    depths: list[int] = []
    node_counts: list[int] = []
    text_lengths: list[int] = []
    token_counts: list[int] = []
    role_counter: Counter = Counter()
    length_hist: defaultdict[int, int] = defaultdict(int)  # bucket → count

    for sample in tqdm(dataset, desc="Computing stats"):
        try:
            tree = json.loads(sample["accessibility"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

        tree = prune_tree(tree)
        if not tree:
            continue

        depths.append(_tree_depth(tree))
        node_counts.append(_node_count(tree))

        text = linearize_tree(tree)
        tlen = len(text)
        text_lengths.append(tlen)
        token_counts.append(len(enc.encode(text)))

        _collect_roles(tree, role_counter)

        # Histogram bucket: every 500 chars
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
        """Return a dict of {pN: value} for each percentile in pcts."""
        if not lst:
            return {f"p{p}": 0 for p in pcts}
        s = sorted(lst)
        n = len(s)
        result: dict[str, int] = {}
        for p in pcts:
            # nearest-rank method
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
# 4. Export
# ---------------------------------------------------------------------------

def export_lava_format(
    dataset,
    output_dir: str | Path,
    max_depth: Optional[int] = None,
    train_ratio: float = 0.85,
) -> None:
    """Export dataset to LLaVA-format JSON for VLM fine-tuning.

    Splits into train/val, saves images to ``{output_dir}/images/``, and
    writes ``train.json`` / ``val.json`` in LLaVA conversation format.
    Also writes a ``config.json`` with all export parameters.

    Parameters
    ----------
    dataset:
        HuggingFace ``Dataset`` object.
    output_dir:
        Root directory for output files.
    max_depth:
        Maximum tree depth for linearization. ``None`` for unlimited.
    train_ratio:
        Fraction of samples to use for training (remainder goes to val).
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    HUMAN_PROMPT = (
        "<image>\n"
        "Generate the complete accessibility tree for this UI screenshot. "
        "Output each element on a new line with indentation showing hierarchy. "
        "Format: ROLE [x,y,w,h] attributes"
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

        linearized = linearize_tree(tree, max_depth=max_depth)
        if not linearized.strip():
            continue

        # Save image
        image = sample["image"]
        img_filename = f"{idx}.png"
        image.save(images_dir / img_filename)

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
        """Remove unpaired surrogate characters that are invalid in UTF-8."""
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
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(
        f"\nExport complete: {len(train_entries)} train / {len(val_entries)} val samples"
        f"\nOutput: {output_dir}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_dataset_split(split: str = "train"):
    """Load the Screen2AX-Tree dataset from HuggingFace."""
    print(f"Loading macpaw-research/Screen2AX-Tree ({split})…")
    return load_dataset("macpaw-research/Screen2AX-Tree", split=split)


def cmd_stats(args: argparse.Namespace) -> None:
    """CLI handler for the ``stats`` subcommand."""
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
    """CLI handler for the ``preview`` subcommand."""
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
        text = linearize_tree(tree, max_depth=max_depth)
        print(f"\n{'='*60}")
        print(f"Sample {shown + 1}  (depth={_tree_depth(tree)}, nodes={_node_count(tree)})")
        print("="*60)
        print(text)
        shown += 1


def cmd_export(args: argparse.Namespace) -> None:
    """CLI handler for the ``export`` subcommand."""
    max_depth: Optional[int] = args.max_depth
    dataset = _load_dataset_split()
    export_lava_format(
        dataset,
        output_dir=args.output_dir,
        max_depth=max_depth,
        train_ratio=args.train_ratio,
    )


def cmd_tokencount(args: argparse.Namespace) -> None:
    """CLI handler for the ``tokencount`` subcommand."""
    enc = tiktoken.get_encoding("cl100k_base")
    dataset = _load_dataset_split()

    # Parse depth values: integers or "None"
    depths: list[Optional[int]] = []
    for d in args.max_depth:
        if d.lower() == "none":
            depths.append(None)
        else:
            depths.append(int(d))

    # Collect trees once
    trees: list[dict] = []
    for sample in tqdm(dataset, desc="Parsing trees"):
        try:
            tree = json.loads(sample["accessibility"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        tree = prune_tree(tree)
        if tree:
            trees.append(tree)

    print(f"\nComparing token counts across {len(trees)} samples\n")
    print(f"{'Depth':<10} {'Avg tokens':>12} {'Min':>8} {'Max':>8} {'Total':>12}")
    print("-" * 54)

    for depth in depths:
        counts = []
        for tree in tqdm(trees, desc=f"depth={depth}", leave=False):
            text = linearize_tree(tree, max_depth=depth)
            counts.append(len(enc.encode(text)))
        avg = sum(counts) / len(counts) if counts else 0
        label = str(depth) if depth is not None else "None"
        print(
            f"{label:<10} {avg:>12.1f} {min(counts):>8} {max(counts):>8} {sum(counts):>12}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen2AX-Tree linearization toolkit for VLM fine-tuning"
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

    # export
    p_export = sub.add_parser("export", help="Export dataset in LLaVA format")
    p_export.add_argument("--output-dir", default="./data_llava", help="Output directory")
    p_export.add_argument(
        "--max-depth", type=lambda x: None if x.lower() == "none" else int(x),
        default=None, metavar="DEPTH",
        help="Max tree depth (integer or 'None')"
    )
    p_export.add_argument("--train-ratio", type=float, default=0.85)

    # tokencount
    p_tok = sub.add_parser("tokencount", help="Compare token counts at different depths")
    p_tok.add_argument(
        "--max-depth", nargs="+", default=["3", "4", "5", "6", "None"],
        metavar="DEPTH",
        help="Space-separated depth values (integers or 'None')"
    )

    args = parser.parse_args()

    dispatch = {
        "stats": cmd_stats,
        "preview": cmd_preview,
        "export": cmd_export,
        "tokencount": cmd_tokencount,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
