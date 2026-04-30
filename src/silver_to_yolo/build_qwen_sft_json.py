"""Build Qwen-VL-Series-Finetune SFT JSON from silver AX tree annotations.

Reads silver linearized .txt files and the train/val split written by
convert.py (silver_to_yolo/convert.py) and emits LLaVA-format JSONs:

  <output-dir>/
    train.json
    val.json
    images/        (symlinks to the source screenshots)

Each sample:
  {
    "id": "<stem>",
    "image": "images/<stem>.png",
    "conversations": [
      {"from": "human", "value": "<image>\\n<USER_PROMPT>"},
      {"from": "gpt",   "value": "<silver ax tree>"}
    ]
  }

The system prompt is NOT embedded per-sample — it lives in the model's
chat template (SYSTEM_MESSAGE in Qwen-VL-Series-Finetune/src/constants.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("silver_to_yolo.build_qwen_sft_json")

IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")

USER_PROMPT = "Generate the complete accessibility tree for this screenshot."


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def build_sample(stem: str, image_rel: str, silver_text: str) -> Dict:
    return {
        "id": stem,
        "image": image_rel,
        "conversations": [
            {"from": "human", "value": f"<image>\n{USER_PROMPT}"},
            {"from": "gpt", "value": silver_text.strip()},
        ],
    }


def load_split(split_path: Path) -> Tuple[List[str], List[str]]:
    data = json.loads(split_path.read_text())
    return list(data["train"]), list(data["val"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Qwen SFT JSONs from silver AX tree annotations."
    )
    p.add_argument("--silver-dir", type=Path, required=True,
                   help="Dir of silver .txt annotations.")
    p.add_argument("--images-dir", type=Path, required=True,
                   help="Dir of screenshot images.")
    p.add_argument("--split-info", type=Path, required=True,
                   help="Path to split_info.json from silver_to_yolo/convert.py.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output root (will contain train.json, val.json, images/).")
    p.add_argument("--min-chars", type=int, default=20,
                   help="Skip silver .txt files shorter than this (probably failed generations).")
    p.add_argument("--max-chars", type=int, default=16000,
                   help="Skip silver .txt files longer than this (runaway generations).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def build_split(
    stems: List[str],
    silver_dir: Path,
    images_dir: Path,
    output_dir: Path,
    min_chars: int,
    max_chars: int,
) -> Tuple[List[Dict], int, int, int]:
    samples: List[Dict] = []
    skipped_missing = 0
    skipped_short = 0
    skipped_long = 0

    for stem in stems:
        silver_path = silver_dir / f"{stem}.txt"
        if not silver_path.exists():
            logger.warning("Missing silver file: %s", silver_path)
            skipped_missing += 1
            continue
        image_path = find_image(images_dir, stem)
        if image_path is None:
            logger.warning("Missing image for stem: %s", stem)
            skipped_missing += 1
            continue
        text = silver_path.read_text(errors="ignore")
        stripped_len = len(text.strip())
        if stripped_len < min_chars:
            logger.debug("Too short (%d chars): %s", stripped_len, stem)
            skipped_short += 1
            continue
        if stripped_len > max_chars:
            logger.info("Runaway generation (%d chars): %s", stripped_len, stem)
            skipped_long += 1
            continue

        image_dst = output_dir / "images" / image_path.name
        link_or_copy(image_path, image_dst)

        samples.append(build_sample(
            stem=stem,
            image_rel=f"images/{image_path.name}",
            silver_text=text,
        ))
    return samples, skipped_missing, skipped_short, skipped_long


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.silver_dir.is_dir():
        raise SystemExit(f"Silver dir not found: {args.silver_dir}")
    if not args.images_dir.is_dir():
        raise SystemExit(f"Images dir not found: {args.images_dir}")
    if not args.split_info.is_file():
        raise SystemExit(f"Split info not found: {args.split_info}")

    train_stems, val_stems = load_split(args.split_info)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Silver dir:   {args.silver_dir}")
    print(f"Images dir:   {args.images_dir}")
    print(f"Split info:   {args.split_info}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Split sizes:  train={len(train_stems)}  val={len(val_stems)}")

    total_missing = 0
    total_short = 0
    total_long = 0
    for split_name, stems in (("train", train_stems), ("val", val_stems)):
        samples, miss, short, long_ = build_split(
            stems, args.silver_dir, args.images_dir, args.output_dir,
            min_chars=args.min_chars, max_chars=args.max_chars,
        )
        total_missing += miss
        total_short += short
        total_long += long_
        out = args.output_dir / f"{split_name}.json"
        out.write_text(json.dumps(samples, ensure_ascii=False, indent=2))
        lengths = [len(s["conversations"][1]["value"]) for s in samples]
        avg = sum(lengths) / len(lengths) if lengths else 0
        print(f"\n[{split_name}] wrote {len(samples)} samples -> {out}")
        print(f"  avg response chars: {avg:.0f}")
        if lengths:
            print(f"  min/max: {min(lengths)} / {max(lengths)}")

    if total_missing or total_short or total_long:
        print(f"\nSkipped: missing={total_missing}  short={total_short}  long(runaway)={total_long}")


if __name__ == "__main__":
    main()
