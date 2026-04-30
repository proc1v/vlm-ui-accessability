#!/usr/bin/env python3
"""
Zero-shot evaluation of ShowUI-2B on Screen2AX-Element dataset.

Runs three prompt strategies:
  A) "List all UI elements" — single prompt per image
  B) "Per-class element detection" — one prompt per class per image
  C) "Element grounding" — ShowUI's native point-prediction task
"""

import argparse
import json
import os
import random
import re
import time
from datetime import datetime

import torch
from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_GROUNDING = (
    "Based on the screenshot of the page, I will give you instructions and you "
    "need to do the following: 1. Return the coordinate of the UI element you "
    "need to click to execute the instruction. 2. The coordinate represents a "
    "clickable location [x, y] for an element, which is a relative coordinate "
    "on the screenshot, scaled from 0 to 1."
)

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1344 * 28 * 28

CLASS_FRIENDLY_NAMES = {
    "AXButton": "buttons",
    "AXDisclosureTriangle": "disclosure triangles (expand/collapse arrows)",
    "AXImage": "images",
    "AXLink": "hyperlinks",
    "AXTextArea": "text input areas",
}

MAX_GROUNDING_QUERIES = 10  # max elements to test per image for Strategy C


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def coco_to_xyxy(bbox):
    """Convert COCO [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def point_in_box(px, py, bbox_xyxy):
    """Check if point (px, py) in absolute pixels is inside bbox [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = bbox_xyxy
    return x1 <= px <= x2 and y1 <= py <= y2


def parse_json_from_text(text):
    """Try to extract a JSON array from model output."""
    # Direct parse
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1)), False
        except json.JSONDecodeError:
            pass
    # Try finding first [ ... ] or { ... }
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        s = text.find(start_char)
        e = text.rfind(end_char)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s : e + 1]), False
            except json.JSONDecodeError:
                pass
    return None, True


def parse_point_from_text(text):
    """Parse [x, y] coordinates from grounding output."""
    try:
        coords = json.loads(text)
        if isinstance(coords, list) and len(coords) == 2:
            return float(coords[0]), float(coords[1])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    match = re.search(r"\[?\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]?", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def build_category_map(ds):
    """Build category id → name mapping from dataset features."""
    try:
        names = ds.features["objects"].feature["category"].names
        return {i: name for i, name in enumerate(names)}
    except Exception:
        # Fallback
        return {
            0: "AXButton",
            1: "AXDisclosureTriangle",
            2: "AXImage",
            3: "AXLink",
            4: "AXTextArea",
        }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference(model, processor, pil_image, system_prompt, query, device, max_new_tokens=128):
    """Run a single ShowUI inference and return the decoded text."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {
                    "type": "image",
                    "image": pil_image,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": query},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    output_text = processor.batch_decode(
        [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def run_strategy_a(model, processor, pil_image, device):
    """Strategy A: list all UI elements in one shot."""
    query = (
        "Based on this screenshot of a macOS application, list all visible UI elements. "
        "For each element, provide:\n"
        "- Type (button, image, text area, link, disclosure triangle)\n"
        "- Approximate bounding box as [x1, y1, x2, y2] in pixel coordinates\n"
        "- Brief description\n\n"
        'Format as JSON array: [{"type": "...", "bbox": [x1, y1, x2, y2], "description": "..."}]'
    )
    system_prompt = (
        "You are a UI analysis assistant. Examine the screenshot carefully and "
        "identify all visible UI elements."
    )
    t0 = time.time()
    raw = run_inference(model, processor, pil_image, system_prompt, query, device, max_new_tokens=2048)
    elapsed = time.time() - t0
    parsed, failed = parse_json_from_text(raw)
    return {
        "raw_output": raw,
        "parsed_elements": parsed if not failed else [],
        "parse_failed": failed,
        "inference_time_s": round(elapsed, 3),
    }


def run_strategy_b(model, processor, pil_image, device, category_map):
    """Strategy B: per-class element detection."""
    system_prompt = (
        "You are a UI analysis assistant. Examine the screenshot carefully."
    )
    per_class = {}
    total_time = 0.0
    for cat_id, cat_name in category_map.items():
        friendly = CLASS_FRIENDLY_NAMES.get(cat_name, cat_name)
        query = (
            f"In this macOS screenshot, locate all {friendly} elements. "
            "For each one, provide its bounding box as [x1, y1, x2, y2] in pixel "
            "coordinates and a brief description. Format as JSON array."
        )
        t0 = time.time()
        raw = run_inference(model, processor, pil_image, system_prompt, query, device, max_new_tokens=2048)
        elapsed = time.time() - t0
        total_time += elapsed
        parsed, failed = parse_json_from_text(raw)
        per_class[cat_name] = {
            "raw_output": raw,
            "parsed_elements": parsed if not failed else [],
            "parse_failed": failed,
            "inference_time_s": round(elapsed, 3),
        }
    return {
        "per_class": per_class,
        "total_inference_time_s": round(total_time, 3),
    }


def run_strategy_c(model, processor, pil_image, device, gt_objects, category_map, img_w, img_h):
    """Strategy C: grounding — predict click point for each GT element."""
    elements = list(zip(gt_objects["bbox"], gt_objects["category"]))
    if len(elements) > MAX_GROUNDING_QUERIES:
        elements = random.sample(elements, MAX_GROUNDING_QUERIES)

    queries = []
    total_time = 0.0
    for bbox_coco, cat_id in elements:
        cat_name = category_map.get(cat_id, f"class_{cat_id}")
        friendly = CLASS_FRIENDLY_NAMES.get(cat_name, cat_name)
        bbox_xyxy = coco_to_xyxy(bbox_coco)
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
        # Build a natural-language query for the element
        region = ""
        if cx < img_w / 3:
            region = "left"
        elif cx > 2 * img_w / 3:
            region = "right"
        else:
            region = "center"
        if cy < img_h / 3:
            region = "top-" + region
        elif cy > 2 * img_h / 3:
            region = "bottom-" + region

        query_text = f"Click on the {friendly.rstrip('s')} in the {region} area of the screen."

        t0 = time.time()
        raw = run_inference(
            model, processor, pil_image, _SYSTEM_GROUNDING, query_text, device, max_new_tokens=128
        )
        elapsed = time.time() - t0
        total_time += elapsed

        px, py = parse_point_from_text(raw)
        hit = False
        predicted_point = None
        if px is not None and py is not None:
            # Convert normalised coords to absolute pixels
            abs_x = px * img_w
            abs_y = py * img_h
            predicted_point = [round(px, 4), round(py, 4)]
            hit = point_in_box(abs_x, abs_y, bbox_xyxy)

        queries.append(
            {
                "query": query_text,
                "gt_bbox_coco": bbox_coco,
                "gt_bbox_xyxy": bbox_xyxy,
                "gt_category": cat_name,
                "raw_output": raw,
                "predicted_point": predicted_point,
                "hit": hit,
            }
        )

    hits = sum(1 for q in queries if q["hit"])
    hit_rate = hits / len(queries) if queries else 0.0
    return {
        "queries": queries,
        "hit_rate": round(hit_rate, 4),
        "hits": hits,
        "total": len(queries),
        "inference_time_s": round(total_time, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zero-shot ShowUI evaluation on Screen2AX-Element")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of images to evaluate")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("Loading Screen2AX-Element dataset...")
    ds = load_dataset("macpaw-research/Screen2AX-Element", split="train")
    print(f"Dataset size: {len(ds)} samples")
    print(f"Features: {ds.features}")

    category_map = build_category_map(ds)
    print(f"Category mapping: {category_map}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\nLoading ShowUI-2B (quantize={args.quantize})...")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
    }
    if args.quantize:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = args.device

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "showlab/ShowUI-2B", **model_kwargs
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    model.eval()
    print("Model loaded.\n")

    # ------------------------------------------------------------------
    # Sample images
    # ------------------------------------------------------------------
    num_samples = min(args.num_samples, len(ds))
    selected_indices = sorted(random.sample(range(len(ds)), num_samples))
    print(f"Selected {num_samples} sample indices: {selected_indices}\n")

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_results = []
    failures = []
    total_start = time.time()

    for i, idx in enumerate(selected_indices):
        sample = ds[idx]
        image_id = sample.get("image_id", idx)
        pil_image = sample["image"].convert("RGB")
        img_w, img_h = pil_image.size
        objects = sample["objects"]

        print(f"[{i + 1}/{num_samples}] Processing image_id={image_id} "
              f"({img_w}x{img_h}, {len(objects['bbox'])} GT elements)...")

        # Save screenshot
        img_path = os.path.join(args.output_dir, f"{image_id}.png")
        pil_image.save(img_path)

        # Build ground truth list
        gt_list = []
        for bbox, cat_id in zip(objects["bbox"], objects["category"]):
            gt_list.append({
                "bbox": bbox,
                "bbox_xyxy": coco_to_xyxy(bbox),
                "category": cat_id,
                "category_name": category_map.get(cat_id, f"class_{cat_id}"),
            })

        result = {
            "image_id": image_id,
            "dataset_index": idx,
            "image_size": [img_w, img_h],
            "ground_truth": gt_list,
            "predictions": {},
        }

        # Strategy A
        try:
            print(f"  Strategy A (list all)...", end=" ", flush=True)
            res_a = run_strategy_a(model, processor, pil_image, args.device)
            result["predictions"]["strategy_a"] = res_a
            n_pred = len(res_a["parsed_elements"]) if res_a["parsed_elements"] else 0
            status = "PARSE_FAILED" if res_a["parse_failed"] else f"{n_pred} elements"
            print(f"done ({res_a['inference_time_s']}s, {status})")
        except Exception as e:
            print(f"FAILED: {e}")
            result["predictions"]["strategy_a"] = {"error": str(e)}
            failures.append(f"image_id={image_id} strategy_a: {e}")

        torch.cuda.empty_cache()

        # Strategy B
        try:
            print(f"  Strategy B (per-class)...", end=" ", flush=True)
            res_b = run_strategy_b(model, processor, pil_image, args.device, category_map)
            result["predictions"]["strategy_b"] = res_b
            print(f"done ({res_b['total_inference_time_s']}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            result["predictions"]["strategy_b"] = {"error": str(e)}
            failures.append(f"image_id={image_id} strategy_b: {e}")

        torch.cuda.empty_cache()

        # Strategy C
        try:
            print(f"  Strategy C (grounding)...", end=" ", flush=True)
            res_c = run_strategy_c(
                model, processor, pil_image, args.device, objects, category_map, img_w, img_h
            )
            result["predictions"]["strategy_c"] = res_c
            print(f"done ({res_c['inference_time_s']}s, "
                  f"hit_rate={res_c['hit_rate']:.2%} [{res_c['hits']}/{res_c['total']}])")
        except Exception as e:
            print(f"FAILED: {e}")
            result["predictions"]["strategy_c"] = {"error": str(e)}
            failures.append(f"image_id={image_id} strategy_c: {e}")

        torch.cuda.empty_cache()

        samples_results.append(result)
        print()

    total_elapsed = time.time() - total_start

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "metadata": {
            "model": "showlab/ShowUI-2B",
            "dataset": "macpaw-research/Screen2AX-Element",
            "num_samples": num_samples,
            "selected_indices": selected_indices,
            "timestamp": timestamp,
            "quantized": args.quantize,
            "seed": args.seed,
            "device": args.device,
        },
        "samples": samples_results,
    }

    out_path = os.path.join(args.output_dir, f"predictions_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total time:        {total_elapsed:.1f}s")
    print(f"Avg time/image:    {total_elapsed / num_samples:.1f}s")
    print(f"Samples processed: {num_samples}")
    print(f"Failures:          {len(failures)}")
    for fail in failures:
        print(f"  - {fail}")
    print(f"Output:            {out_path}")


if __name__ == "__main__":
    main()
