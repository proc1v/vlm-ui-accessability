#!/usr/bin/env python3
"""Batch-run OmniParser perception over a directory of images.

For each image, writes a Screen2AX-Task-compatible JSON file to --output-dir,
shaped as a flat forest of nodes:
    [{"cls": "AXButton", "box": [x1,y1,x2,y2], "value": "..."}, ...]

Coords are emitted normalized to 0–1000 so that
`eval_screen2ax_task.py --pred-format screen2ax_json --pred-coords normalized`
denormalizes them correctly against each annotation's image_width/image_height.

Cross-image Florence-2 batching: we pre-compute OCR + YOLO + overlap filtering
for a chunk of images (CPU-bound), pool *all* their icon crops into one buffer,
then run Florence in `--batch-size` mini-batches. This keeps the GPU saturated
even when individual images have far fewer than `batch_size` icons.

Usage:
    python run_batch.py \
        --images-dir /workspace/data/screen2ax_task/images \
        --output-dir /workspace/inference_results_omniparser \
        --device cuda --batch-size 256 --chunk-size 32
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*num_beams.*early_stopping.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util.utils import (
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model,
    int_box_area,
    predict_yolo,
    remove_overlap_new,
)


TYPE_TO_AX = {"text": "AXStaticText", "icon": "AXButton"}
TO_PIL = ToPILImage()


# ---------------------------------------------------------------------------
# Output mapping
# ---------------------------------------------------------------------------

def to_screen2ax_json(parsed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """parsed_content_list (normalized 0–1) -> flat Screen2AX nodes (0–1000)."""
    out = []
    for item in parsed:
        bbox = item.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = bbox
        node = {
            "cls": TYPE_TO_AX.get(item.get("type", ""), "AXUnknown"),
            "box": [
                int(round(x1 * 1000)), int(round(y1 * 1000)),
                int(round(x2 * 1000)), int(round(y2 * 1000)),
            ],
        }
        content = item.get("content")
        if content:
            node["value"] = str(content)
        out.append(node)
    return out


# ---------------------------------------------------------------------------
# Stage A — per-image perception (no captioning)
# ---------------------------------------------------------------------------

def stage_a_perception(
    img_path: Path, som_model, args
) -> Optional[Tuple[List[Dict[str, Any]], List[Image.Image]]]:
    """Run OCR + YOLO + overlap filter for a single image.

    Returns (filtered_boxes_elem, icon_crops) where icon_crops is the
    list of 64x64 PIL crops corresponding to elements with content=None
    (i.e. detected icons that need Florence captioning), in the same order
    as their occurrence in filtered_boxes_elem.
    """
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    (text, ocr_bbox), _ = check_ocr_box(
        str(img_path),
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=args.use_paddleocr,
    )

    xyxy, _, _ = predict_yolo(
        model=som_model, image=image, box_threshold=args.box_threshold,
        imgsz=(h, w), scale_img=False, iou_threshold=0.1,
    )
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    image_np = np.asarray(image)

    if ocr_bbox:
        ocr_bbox_t = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox_norm = ocr_bbox_t.tolist()
    else:
        ocr_bbox_norm = None

    ocr_bbox_elem = [
        {"type": "text", "bbox": box, "interactivity": False,
         "content": txt, "source": "box_ocr_content_ocr"}
        for box, txt in zip(ocr_bbox_norm or [], text)
        if int_box_area(box, w, h) > 0
    ]
    xyxy_elem = [
        {"type": "icon", "bbox": box, "interactivity": True, "content": None}
        for box in xyxy.tolist() if int_box_area(box, w, h) > 0
    ]
    filtered = remove_overlap_new(
        boxes=xyxy_elem, iou_threshold=args.iou_threshold, ocr_bbox=ocr_bbox_elem,
    )
    # Keep elements with text-content first, icons (content=None) at the end —
    # mirrors get_som_labeled_img so captions slot back in the same order.
    filtered_boxes_elem = sorted(filtered, key=lambda x: x["content"] is None)

    # Crop each icon (content=None) to 64x64 for Florence.
    crops: List[Image.Image] = []
    for elem in filtered_boxes_elem:
        if elem["content"] is not None:
            continue
        x1, y1, x2, y2 = elem["bbox"]
        xmin = int(x1 * image_np.shape[1]); xmax = int(x2 * image_np.shape[1])
        ymin = int(y1 * image_np.shape[0]); ymax = int(y2 * image_np.shape[0])
        try:
            cropped = image_np[ymin:ymax, xmin:xmax, :]
            cropped = cv2.resize(cropped, (64, 64))
            crops.append(TO_PIL(cropped))
        except Exception:
            crops.append(None)  # signal failure; we'll fill with empty caption later

    return filtered_boxes_elem, crops


# ---------------------------------------------------------------------------
# Stage B — cross-image Florence batching
# ---------------------------------------------------------------------------

def stage_b_caption_batch(
    crops: List[Optional[Image.Image]], caption_model_processor, batch_size: int,
) -> List[str]:
    """Caption a pooled buffer of crops in mini-batches of `batch_size`."""
    if not crops:
        return []
    model = caption_model_processor["model"]
    processor = caption_model_processor["processor"]
    device = model.device
    is_florence = "florence" in model.config.name_or_path

    # Mask out failed crops (None) — caption "" for them.
    valid_idx = [i for i, c in enumerate(crops) if c is not None]
    valid_crops = [crops[i] for i in valid_idx]
    captions_valid: List[str] = []

    prompt = "<CAPTION>" if is_florence else "The image shows"
    for i in range(0, len(valid_crops), batch_size):
        batch = valid_crops[i : i + batch_size]
        if device.type == "cuda":
            inputs = processor(
                images=batch, text=[prompt] * len(batch),
                return_tensors="pt", do_resize=False,
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(
                images=batch, text=[prompt] * len(batch), return_tensors="pt",
            ).to(device=device)
        if is_florence:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20, num_beams=1, do_sample=False,
            )
        else:
            generated_ids = model.generate(
                **inputs, max_length=100, num_beams=5,
                no_repeat_ngram_size=2, early_stopping=True,
                num_return_sequences=1,
            )
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        captions_valid.extend(d.strip() for d in decoded)

    captions = [""] * len(crops)
    for j, idx in enumerate(valid_idx):
        captions[idx] = captions_valid[j]
    return captions


# ---------------------------------------------------------------------------
# Chunk runner
# ---------------------------------------------------------------------------

def process_chunk(
    img_paths: List[Path], som_model, caption_model_processor, args,
) -> List[Tuple[Path, Optional[List[Dict[str, Any]]], Optional[Exception]]]:
    """Run Stage A on each image in chunk, then Stage B on the pooled crops."""
    perception: List[Tuple[Path, Optional[Tuple[List, List]], Optional[Exception]]] = []
    for p in img_paths:
        try:
            res = stage_a_perception(p, som_model, args)
            perception.append((p, res, None))
        except Exception as e:
            perception.append((p, None, e))

    # Pool all crops, remembering (img_idx, slot_in_image)
    pool: List[Optional[Image.Image]] = []
    crop_origin: List[Tuple[int, int]] = []  # (perception index, slot in that image's crops)
    for i, (_, res, err) in enumerate(perception):
        if err is not None or res is None:
            continue
        _, crops = res
        for j, c in enumerate(crops):
            crop_origin.append((i, j))
            pool.append(c)

    if caption_model_processor is not None and pool:
        captions = stage_b_caption_batch(pool, caption_model_processor, args.batch_size)
    else:
        captions = [""] * len(pool)

    # Scatter captions back into per-image filtered_boxes_elem
    per_image_caps: Dict[int, List[str]] = {}
    for caption, (img_i, _slot) in zip(captions, crop_origin):
        per_image_caps.setdefault(img_i, []).append(caption)

    out: List[Tuple[Path, Optional[List[Dict[str, Any]]], Optional[Exception]]] = []
    for i, (path, res, err) in enumerate(perception):
        if err is not None or res is None:
            out.append((path, None, err))
            continue
        filtered_boxes_elem, _ = res
        caps = per_image_caps.get(i, [])
        cap_iter = iter(caps)
        for elem in filtered_boxes_elem:
            if elem["content"] is None:
                elem["content"] = next(cap_iter, "")
        out.append((path, filtered_boxes_elem, None))
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--yolo-weights", default="weights/icon_detect/model.pt")
    p.add_argument("--caption-weights", default="weights/icon_caption_florence")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--box-threshold", type=float, default=0.05)
    p.add_argument("--iou-threshold", type=float, default=0.7)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Florence-2 mini-batch (crops per forward pass)")
    p.add_argument("--chunk-size", type=int, default=32,
                   help="Number of images whose crops are pooled together for "
                        "captioning. Larger = better GPU utilization, more RAM. "
                        "Each image yields ~30-50 icon crops.")
    p.add_argument("--use-paddleocr", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-semantics", action="store_true",
                   help="Skip Florence-2 captioning")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        [p for p in images_dir.iterdir()
         if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if args.limit is not None:
        images = images[: args.limit]

    if args.resume:
        images = [p for p in images if not (out_dir / f"{p.stem}.json").exists()]

    print(f"loading YOLO from {args.yolo_weights} on {args.device}")
    som_model = get_yolo_model(args.yolo_weights).to(args.device)
    caption_model_processor = None
    if not args.no_semantics:
        print(f"loading Florence-2 from {args.caption_weights}")
        caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path=args.caption_weights,
            device=args.device,
        )

    n_done = n_failed = 0
    pbar = tqdm(total=len(images), desc="omniparser", unit="img")
    for start in range(0, len(images), args.chunk_size):
        chunk = images[start : start + args.chunk_size]
        results = process_chunk(chunk, som_model, caption_model_processor, args)
        for path, elems, err in results:
            sample_id = path.stem
            if err is not None or elems is None:
                n_failed += 1
                tqdm.write(f"[{sample_id}] FAILED: {type(err).__name__}: {err}")
            else:
                nodes = to_screen2ax_json(elems)
                (out_dir / f"{sample_id}.json").write_text(
                    json.dumps(nodes, ensure_ascii=False), encoding="utf-8",
                )
                n_done += 1
            pbar.update(1)
            pbar.set_postfix(done=n_done, failed=n_failed)
    pbar.close()
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
