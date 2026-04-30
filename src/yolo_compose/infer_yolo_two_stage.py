"""Run the two trained YOLOv11 detectors on a split and cache detections.

Produces one JSON per image in --detections-dir:
  {
    "width": <px>, "height": <px>,
    "leaves":  [{"role": "AXButton", "bbox": [x1,y1,x2,y2], "score": 0.87}, ...],
    "groups":  [{"bbox": [x1,y1,x2,y2], "score": 0.74}, ...],
    "stage1_conf": 0.25, "stage2_conf": 0.25, "nms_iou": 0.45, "max_det": 300
  }

Coordinates are in the 0-1000 silver space (not pixels). Composition reads
these JSONs and is decoupled from inference — retune nesting rules for free.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("yolo_compose.infer")

STAGE1_NAMES = [
    "AXButton",
    "AXStaticText",
    "AXImage",
    "AXLink",
    "AXTextArea",
    "AXDisclosureTriangle",
]
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def xyxyn_to_0_1000(xyxyn: Tuple[float, float, float, float]) -> List[int]:
    x1, y1, x2, y2 = (float(v) for v in xyxyn)
    out = [
        max(0, min(1000, int(round(x1 * 1000)))),
        max(0, min(1000, int(round(y1 * 1000)))),
        max(0, min(1000, int(round(x2 * 1000)))),
        max(0, min(1000, int(round(y2 * 1000)))),
    ]
    return out


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_stems(split_info: Path, split: str) -> List[str]:
    data = json.loads(split_info.read_text())
    if split not in ("train", "val"):
        raise SystemExit(f"--split must be train|val, got {split}")
    return list(data[split])


def main() -> None:
    from ultralytics import YOLO
    from tqdm import tqdm
    from PIL import Image

    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    images_dir = Path(args.images_dir)
    out_dir = Path(args.detections_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = load_stems(Path(args.split_info), args.split)
    if args.num_samples is not None:
        stems = stems[: args.num_samples]
    logger.info("split=%s  n=%d", args.split, len(stems))

    logger.info("Loading Stage 1 (elements): %s", args.stage1_weights)
    stage1 = YOLO(args.stage1_weights)
    logger.info("Loading Stage 2 (groups):   %s", args.stage2_weights)
    stage2 = YOLO(args.stage2_weights)

    device = args.device

    n_ok = 0
    n_missing = 0
    total_latency = 0.0
    total_leaves = 0
    total_groups = 0

    for stem in tqdm(stems, desc=f"yolo[{args.split}]"):
        out_path = out_dir / f"{stem}.json"
        if args.resume and out_path.exists():
            continue

        img_path = find_image(images_dir, stem)
        if img_path is None:
            logger.warning("Missing image for %s", stem)
            n_missing += 1
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        t0 = time.time()
        r1 = stage1.predict(
            source=str(img_path),
            conf=args.stage1_conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )[0]
        r2 = stage2.predict(
            source=str(img_path),
            conf=args.stage2_conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )[0]
        latency = time.time() - t0

        leaves: List[Dict] = []
        if r1.boxes is not None and len(r1.boxes):
            xyxyn = r1.boxes.xyxyn.cpu().tolist()
            cls = r1.boxes.cls.cpu().tolist()
            conf = r1.boxes.conf.cpu().tolist()
            for b, c, s in zip(xyxyn, cls, conf):
                ci = int(c)
                if ci < 0 or ci >= len(STAGE1_NAMES):
                    continue
                leaves.append({
                    "role": STAGE1_NAMES[ci],
                    "bbox": xyxyn_to_0_1000(b),
                    "score": float(s),
                })

        groups: List[Dict] = []
        if r2.boxes is not None and len(r2.boxes):
            xyxyn = r2.boxes.xyxyn.cpu().tolist()
            conf = r2.boxes.conf.cpu().tolist()
            for b, s in zip(xyxyn, conf):
                groups.append({
                    "bbox": xyxyn_to_0_1000(b),
                    "score": float(s),
                })

        out_path.write_text(json.dumps({
            "stem": stem,
            "image": img_path.name,
            "width": img_w,
            "height": img_h,
            "leaves": leaves,
            "groups": groups,
            "latency_seconds": latency,
            "stage1_conf": args.stage1_conf,
            "stage2_conf": args.stage2_conf,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "imgsz": args.imgsz,
        }, ensure_ascii=False))

        n_ok += 1
        total_latency += latency
        total_leaves += len(leaves)
        total_groups += len(groups)

    avg_lat = total_latency / n_ok if n_ok else 0.0
    print(f"\n=== YOLO two-stage inference [{args.split}] ===")
    print(f"Images processed:    {n_ok}")
    print(f"Missing:             {n_missing}")
    print(f"Avg latency:         {avg_lat:.3f}s")
    print(f"Avg leaves/image:    {total_leaves / n_ok if n_ok else 0:.1f}")
    print(f"Avg groups/image:    {total_groups / n_ok if n_ok else 0:.1f}")
    print(f"Detections dir:      {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--stage1-weights", required=True,
                   help="Path to stage1_elements best.pt")
    p.add_argument("--stage2-weights", required=True,
                   help="Path to stage2_groups best.pt")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--split-info", required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--detections-dir", required=True)
    p.add_argument("--stage1-conf", type=float, default=0.25)
    p.add_argument("--stage2-conf", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=500)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
