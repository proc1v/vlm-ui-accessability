#!/usr/bin/env python3
"""Dump MacPaw/Screen2AX-Task from HuggingFace to a directory layout that
inference.py / evaluate.py already understand.

Output layout:
    {output-dir}/
        images/{id}.png                # one PNG per sample
        annotations/{id}.json          # sidecar with command, gt_box, dims, etc.
        dataset_meta.json              # summary (n_samples, hf_id, split)

Stem naming uses the HuggingFace row index (0..N-1) without zero-padding.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

HF_ID = "MacPaw/Screen2AX-Task"
DEFAULT_SPLIT = "train"


def dump(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    ann_dir = out_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {args.hf_id} (split={args.split})")
    ds = load_dataset(args.hf_id, split=args.split)
    n = len(ds)
    if args.num_samples is not None:
        n = min(n, args.num_samples)
    logger.info(f"Will dump {n} samples to {out_dir}")

    written = 0
    skipped = 0
    for idx in tqdm(range(n), desc="dump"):
        stem = str(idx)
        img_path = images_dir / f"{stem}.png"
        ann_path = ann_dir / f"{stem}.json"

        if args.resume and img_path.exists() and ann_path.exists():
            skipped += 1
            continue

        sample = ds[idx]
        image = sample["image"]
        # Ensure RGB (some HF images may be RGBA / palette)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(img_path, format="PNG")

        x1 = float(sample["x1"]); y1 = float(sample["y1"])
        x2 = float(sample["x2"]); y2 = float(sample["y2"])
        ann = {
            "id": idx,
            "command": sample["command"],
            "visual_description": sample["visual_description"],
            "gt_box": [x1, y1, x2, y2],          # pixel coords
            "image_width": int(sample["image_width"]),
            "image_height": int(sample["image_height"]),
        }
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False)
        written += 1

    meta = {
        "hf_id": args.hf_id,
        "split": args.split,
        "n_samples": n,
        "written": written,
        "skipped_existing": skipped,
        "stem_format": "row_index",
    }
    with open(out_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Done. Wrote {written}, skipped {skipped}. Output: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dump MacPaw/Screen2AX-Task to disk in inference.py layout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="/workspace/data/screen2ax_task",
                   help="Target directory; images/ and annotations/ will be created inside")
    p.add_argument("--hf-id", default=HF_ID)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limit for smoke testing")
    p.add_argument("--resume", action="store_true",
                   help="Skip stems whose image+annotation both already exist")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    dump(args)
