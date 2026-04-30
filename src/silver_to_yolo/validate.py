"""Validate a YOLO dataset produced by convert.py.

Checks image/label pairing, coordinate ranges, class IDs, aspect ratios,
class distribution, and (optionally) draws bboxes for a random sample.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("silver_to_yolo.validate")

STAGE1_CLASSES = [
    "AXButton",
    "AXStaticText",
    "AXImage",
    "AXLink",
    "AXTextArea",
    "AXDisclosureTriangle",
]
STAGE2_CLASSES = ["AXGroup"]

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def parse_dataset_yaml(path: Path) -> Tuple[int, List[str]]:
    """Minimal YAML parser for the subset we write."""
    nc = 0
    names: List[str] = []
    in_names = False
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("nc:"):
            nc = int(line.split(":", 1)[1].strip())
            in_names = False
        elif line.startswith("names:"):
            in_names = True
        elif in_names and line.startswith(" "):
            # "  0: AXButton"
            _, name = line.split(":", 1)
            names.append(name.strip())
        else:
            in_names = False
    return nc, names


def find_image_for_label(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def check_split(
    dataset_dir: Path, split: str, num_classes: int
) -> dict:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    stats = {
        "split": split,
        "image_count": 0,
        "label_count": 0,
        "missing_labels": [],
        "missing_images": [],
        "empty_labels": [],
        "bad_coords": [],
        "bad_class_ids": [],
        "box_count": 0,
        "class_counter": Counter(),
        "aspect_outliers": [],
    }

    image_stems = {p.stem for p in images_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS}
    label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    stats["image_count"] = len(image_stems)
    stats["label_count"] = len(label_stems)

    for stem in sorted(image_stems - label_stems):
        stats["missing_labels"].append(stem)
    for stem in sorted(label_stems - image_stems):
        stats["missing_images"].append(stem)

    for lbl in sorted(labels_dir.glob("*.txt")):
        lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
        if not lines:
            stats["empty_labels"].append(lbl.stem)
            continue
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                stats["bad_coords"].append(f"{lbl.name}: {ln!r}")
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = (float(x) for x in parts[1:])
            except ValueError:
                stats["bad_coords"].append(f"{lbl.name}: {ln!r}")
                continue
            if cls_id < 0 or cls_id >= num_classes:
                stats["bad_class_ids"].append(f"{lbl.name}: {cls_id}")
                continue
            if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
                stats["bad_coords"].append(f"{lbl.name}: {ln}")
                continue
            if w <= 0 or h <= 0:
                stats["bad_coords"].append(f"{lbl.name}: {ln}")
                continue
            ratio = max(w, h) / max(1e-9, min(w, h))
            if ratio > 50.0:
                stats["aspect_outliers"].append(
                    f"{lbl.name}: ratio={ratio:.1f} (w={w:.3f}, h={h:.3f})"
                )
            stats["box_count"] += 1
            stats["class_counter"][cls_id] += 1
    return stats


def visualize(
    dataset_dir: Path,
    class_names: List[str],
    n: int,
    seed: int,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.error("Pillow is required for --visualize. Install with: pip install pillow")
        return

    out_dir = dataset_dir.parent / "debug_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather candidates from both splits.
    samples: List[Tuple[Path, Path]] = []
    for split in ("train", "val"):
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        for img in images_dir.glob("*"):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = labels_dir / f"{img.stem}.txt"
            if lbl.exists():
                samples.append((img, lbl))
    rng = random.Random(seed)
    rng.shuffle(samples)
    samples = samples[:n]

    palette = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (210, 245, 60), (250, 190, 212),
    ]

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for img_path, lbl_path in samples:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        W, H = img.size
        for ln in lbl_path.read_text().splitlines():
            parts = ln.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = (float(x) for x in parts[1:])
            x1 = (xc - w / 2) * W
            y1 = (yc - h / 2) * H
            x2 = (xc + w / 2) * W
            y2 = (yc + h / 2) * H
            color = palette[cls_id % len(palette)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            if font:
                draw.text((x1 + 2, y1 + 2), label, fill=color, font=font)
        out_path = out_dir / f"{dataset_dir.name}_{img_path.stem}.jpg"
        img.save(out_path, quality=90)
        logger.info("Wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, required=True,
                   help="Path to e.g. yolo/stage1_detection")
    p.add_argument("--visualize", type=int, default=0,
                   help="Number of random samples to visualize (default 0).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    dataset_dir: Path = args.dataset_dir
    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"Missing {yaml_path}")

    nc, class_names = parse_dataset_yaml(yaml_path)
    print(f"Dataset: {dataset_dir}")
    print(f"Classes ({nc}): {class_names}")

    any_errors = False
    for split in ("train", "val"):
        stats = check_split(dataset_dir, split, nc)
        print(f"\n[{split}]")
        print(f"  Images: {stats['image_count']}  Labels: {stats['label_count']}")
        print(f"  Boxes:  {stats['box_count']}")
        for cls_id in range(nc):
            c = stats["class_counter"].get(cls_id, 0)
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            print(f"    {cls_id} {name:<22} {c}")
        for field in ("missing_labels", "missing_images", "empty_labels",
                      "bad_coords", "bad_class_ids", "aspect_outliers"):
            items = stats[field]
            if items:
                any_errors = True
                print(f"  {field}: {len(items)}")
                for x in items[:5]:
                    print(f"    - {x}")
                if len(items) > 5:
                    print(f"    ... (+{len(items) - 5} more)")

    if args.visualize > 0:
        visualize(dataset_dir, class_names, args.visualize, args.seed)

    if any_errors:
        print("\nValidation completed with warnings.")
        sys.exit(1)
    print("\nValidation OK.")


if __name__ == "__main__":
    main()
