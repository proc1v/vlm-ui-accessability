"""Convert linearized silver AX tree annotations to YOLO detection format.

Produces two parallel YOLO datasets from the same silver annotations:
  Stage 1 (element detection): leaf non-AXGroup nodes, 6 classes.
  Stage 2 (group detection):   AXGroup nodes that have >=1 child, 1 class.

Both stages share the same train/val split so they stay aligned for downstream
hybrid evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Reuse the existing parser so we stay consistent with the eval pipeline.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "screen2ax_eval"))
from utils.parser import AXNode, parse_tree  # noqa: E402

logger = logging.getLogger("silver_to_yolo.convert")

STAGE1_CLASSES: List[str] = [
    "AXButton",
    "AXStaticText",
    "AXImage",
    "AXLink",
    "AXTextArea",
    "AXDisclosureTriangle",
]
STAGE1_CLASS_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(STAGE1_CLASSES)}

STAGE2_CLASSES: List[str] = ["AXGroup"]
STAGE2_CLASS_TO_ID: Dict[str, int] = {"AXGroup": 0}

IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------------------
# Geometry


def silver_bbox_to_yolo(bbox_1000: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert [x1,y1,x2,y2] in 0-1000 range to YOLO (xc,yc,w,h) in 0-1.

    Coordinates are clamped to [0, 1000] before conversion to absorb minor
    out-of-bounds predictions from the silver generator.
    """
    x1, y1, x2, y2 = bbox_1000
    x1 = max(0.0, min(1000.0, float(x1)))
    y1 = max(0.0, min(1000.0, float(y1)))
    x2 = max(0.0, min(1000.0, float(x2)))
    y2 = max(0.0, min(1000.0, float(y2)))

    x_center = (x1 + x2) / 2.0 / 1000.0
    y_center = (y1 + y2) / 2.0 / 1000.0
    width = (x2 - x1) / 1000.0
    height = (y2 - y1) / 1000.0

    clamp = lambda v: max(0.0, min(1.0, v))  # noqa: E731
    return clamp(x_center), clamp(y_center), clamp(width), clamp(height)


def is_degenerate(bbox_1000: Sequence[float]) -> bool:
    x1, y1, x2, y2 = bbox_1000
    return x2 <= x1 or y2 <= y1


# ---------------------------------------------------------------------------
# Annotation extraction


def collect_stage1(nodes: List[AXNode]) -> List[AXNode]:
    """Leaf nodes whose role is a Stage-1 class."""
    out = []
    for n in nodes:
        if n.children:
            continue
        if n.role == "AXGroup":
            continue
        if n.role not in STAGE1_CLASS_TO_ID:
            continue
        if n.bbox is None:
            continue
        out.append(n)
    return out


def collect_stage2(nodes: List[AXNode]) -> List[AXNode]:
    """AXGroup nodes with >=1 child."""
    out = []
    for n in nodes:
        if n.role != "AXGroup":
            continue
        if not n.children:
            continue
        if n.bbox is None:
            continue
        out.append(n)
    return out


def tree_max_depth(nodes: List[AXNode]) -> int:
    return max((n.depth for n in nodes), default=0)


# ---------------------------------------------------------------------------
# Dataset building


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def write_label_file(
    path: Path,
    boxes: List[Tuple[int, float, float, float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for cls_id, xc, yc, w, h in boxes:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError as e:
        logger.debug("Symlink failed (%s); falling back to copy for %s", e, src.name)
        shutil.copy2(src, dst)


def write_dataset_yaml(root: Path, class_names: List[str]) -> None:
    # Write YAML manually to avoid a PyYAML dependency.
    lines = [
        f"path: {root.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    (root / "dataset.yaml").write_text("\n".join(lines) + "\n")


def split_files(
    stems: List[str],
    val_ratio: float,
    seed: int,
    split_by_app: bool,
    app_delimiter: str,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)

    if split_by_app:
        groups: Dict[str, List[str]] = {}
        for s in stems:
            app = s.split(app_delimiter, 1)[0] if app_delimiter in s else s
            groups.setdefault(app, []).append(s)
        app_keys = sorted(groups.keys())
        rng.shuffle(app_keys)
        n_val_apps = max(1, int(round(len(app_keys) * val_ratio)))
        val_apps = set(app_keys[:n_val_apps])
        train, val = [], []
        for app, items in groups.items():
            (val if app in val_apps else train).extend(sorted(items))
        return sorted(train), sorted(val)

    shuffled = list(stems)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio))) if shuffled else 0
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


# ---------------------------------------------------------------------------
# Per-file processing


def extract_boxes_for_file(
    silver_path: Path,
    stage: int,
    min_bbox_size: float,
) -> Tuple[List[Tuple[int, float, float, float, float]], int]:
    """Return (yolo_boxes, skipped_count) for one silver file."""
    text = silver_path.read_text(errors="ignore")
    nodes = parse_tree(text)

    if stage == 1:
        candidates = collect_stage1(nodes)
        class_to_id = STAGE1_CLASS_TO_ID
    else:
        candidates = collect_stage2(nodes)
        class_to_id = STAGE2_CLASS_TO_ID

    boxes: List[Tuple[int, float, float, float, float]] = []
    skipped = 0
    for node in candidates:
        assert node.bbox is not None
        if is_degenerate(node.bbox):
            skipped += 1
            logger.debug("Degenerate bbox in %s: %s", silver_path.name, node.bbox)
            continue
        xc, yc, w, h = silver_bbox_to_yolo(node.bbox)
        if w <= min_bbox_size or h <= min_bbox_size:
            skipped += 1
            logger.debug(
                "Tiny bbox in %s: w=%.4f h=%.4f (role=%s)",
                silver_path.name,
                w,
                h,
                node.role,
            )
            continue
        cls_id = class_to_id[node.role]
        boxes.append((cls_id, xc, yc, w, h))
    return boxes, skipped


# ---------------------------------------------------------------------------
# Stage writing


def build_stage(
    stage: int,
    output_root: Path,
    silver_dir: Path,
    images_dir: Path,
    train_stems: List[str],
    val_stems: List[str],
    min_bbox_size: float,
    skip_empty: bool,
) -> Dict:
    stage_name = "stage1_detection" if stage == 1 else "stage2_grouping"
    class_names = STAGE1_CLASSES if stage == 1 else STAGE2_CLASSES
    stage_root = output_root / stage_name
    (stage_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (stage_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (stage_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (stage_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    class_counter: Counter = Counter()
    total_boxes = 0
    total_skipped_degen = 0
    total_images = 0
    empty_images = 0
    depth_sum = 0
    depth_count = 0
    groups_with_no_children = 0  # only meaningful for stage 2

    for split, stems in (("train", train_stems), ("val", val_stems)):
        for stem in stems:
            silver_path = silver_dir / f"{stem}.txt"
            if not silver_path.exists():
                logger.warning("Missing silver file: %s", silver_path)
                continue
            image_path = find_image(images_dir, stem)
            if image_path is None:
                logger.warning("Missing image for stem: %s", stem)
                continue

            boxes, skipped = extract_boxes_for_file(
                silver_path, stage, min_bbox_size
            )
            total_skipped_degen += skipped

            if stage == 2:
                nodes = parse_tree(silver_path.read_text(errors="ignore"))
                groups_with_no_children += sum(
                    1 for n in nodes if n.role == "AXGroup" and not n.children
                )
                md = tree_max_depth(nodes)
                depth_sum += md
                depth_count += 1

            if not boxes:
                empty_images += 1
                if skip_empty:
                    continue

            label_path = stage_root / "labels" / split / f"{stem}.txt"
            write_label_file(label_path, boxes)

            image_dst = stage_root / "images" / split / image_path.name
            link_or_copy(image_path, image_dst)

            total_images += 1
            total_boxes += len(boxes)
            for cls_id, *_ in boxes:
                class_counter[class_names[cls_id]] += 1

    write_dataset_yaml(stage_root, class_names)

    stats: Dict = {
        "total_images": total_images,
        "total_boxes": total_boxes,
        "skipped_degenerate": total_skipped_degen,
        "empty_images": empty_images,
        "per_class": dict(class_counter),
    }
    if stage == 2:
        stats["groups_with_no_children"] = groups_with_no_children
        stats["avg_nesting_depth"] = (
            depth_sum / depth_count if depth_count else 0.0
        )
    return stats


# ---------------------------------------------------------------------------
# Reporting


def print_stage_report(stage: int, stats: Dict) -> None:
    class_names = STAGE1_CLASSES if stage == 1 else STAGE2_CLASSES
    label = "Stage 1 (Element Detection)" if stage == 1 else "Stage 2 (Group Detection)"
    print(f"\n{label}:")
    print(f"  Images written:        {stats['total_images']}")
    print(f"  Total boxes:           {stats['total_boxes']:,}")
    if stats["total_images"]:
        avg = stats["total_boxes"] / stats["total_images"]
        key = "Avg boxes per image" if stage == 1 else "Avg groups per image"
        print(f"  {key}:   {avg:.1f}")
    if stage == 2 and "avg_nesting_depth" in stats:
        print(f"  Avg nesting depth:     {stats['avg_nesting_depth']:.1f}")
    print("  Per-class distribution:")
    total = stats["total_boxes"] or 1
    for name in class_names:
        c = stats["per_class"].get(name, 0)
        print(f"    {name:<22} {c:>8,} ({100.0 * c / total:.1f}%)")
    print(f"  Skipped (degenerate):  {stats['skipped_degenerate']}")
    print(f"  Empty images:          {stats['empty_images']}")
    if stage == 2 and "groups_with_no_children" in stats:
        print(f"  Skipped (no children): {stats['groups_with_no_children']}")


# ---------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert silver AX tree annotations to YOLO format."
    )
    p.add_argument("--silver-dir", type=Path, required=True,
                   help="Directory of silver .txt annotations.")
    p.add_argument("--images-dir", type=Path, required=True,
                   help="Directory of screenshot images (PNG/JPG).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Root output directory for YOLO datasets.")
    p.add_argument("--stage", default="both", choices=["1", "2", "both"],
                   help="Which stage(s) to generate.")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-by-app", action="store_true",
                   help="Split by app prefix to avoid data leakage.")
    p.add_argument("--app-delimiter", default="_",
                   help="Delimiter separating app name from image ID.")
    p.add_argument("--min-bbox-size", type=float, default=0.001,
                   help="Min bbox width/height (normalized) to include.")
    p.add_argument("--skip-empty", dest="skip_empty", action="store_true", default=True)
    p.add_argument("--no-skip-empty", dest="skip_empty", action="store_false")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    silver_dir: Path = args.silver_dir
    images_dir: Path = args.images_dir
    output_dir: Path = args.output_dir

    if not silver_dir.is_dir():
        raise SystemExit(f"Silver dir not found: {silver_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"Images dir not found: {images_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover stems that have both an annotation and an image.
    all_stems: List[str] = []
    missing_images = 0
    for f in sorted(silver_dir.glob("*.txt")):
        stem = f.stem
        if find_image(images_dir, stem) is None:
            missing_images += 1
            continue
        all_stems.append(stem)
    if not all_stems:
        raise SystemExit("No silver/image pairs found.")
    if missing_images:
        logger.warning("%d silver files had no matching image", missing_images)

    train_stems, val_stems = split_files(
        all_stems,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_by_app=args.split_by_app,
        app_delimiter=args.app_delimiter,
    )

    split_info = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "split_by_app": args.split_by_app,
        "app_delimiter": args.app_delimiter,
        "total": len(all_stems),
        "train_count": len(train_stems),
        "val_count": len(val_stems),
        "train": train_stems,
        "val": val_stems,
    }
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))

    print("=== Silver -> YOLO Conversion ===")
    print(f"Silver dir: {silver_dir}")
    print(f"Images dir: {images_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Total annotations: {len(all_stems)}")
    print(f"Train: {len(train_stems)} | Val: {len(val_stems)}")

    stages_to_run: List[int] = []
    if args.stage in ("1", "both"):
        stages_to_run.append(1)
    if args.stage in ("2", "both"):
        stages_to_run.append(2)

    for stage in stages_to_run:
        stats = build_stage(
            stage=stage,
            output_root=output_dir,
            silver_dir=silver_dir,
            images_dir=images_dir,
            train_stems=train_stems,
            val_stems=val_stems,
            min_bbox_size=args.min_bbox_size,
            skip_empty=args.skip_empty,
        )
        print_stage_report(stage, stats)


if __name__ == "__main__":
    main()
