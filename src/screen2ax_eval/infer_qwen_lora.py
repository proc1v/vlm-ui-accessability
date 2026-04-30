#!/usr/bin/env python3
"""Local HF inference for the Qwen3-VL-8B LoRA fine-tune on Screen2AX.

Produces the same on-disk layout as inference.py (raw/ parsed/ metadata/),
so evaluate.py can be pointed straight at parsed/ without changes.

Usage:
    python infer_qwen_lora.py \\
        --lora-dir /workspace/trained_models/qwen3vl_8b_screen2ax_lora \\
        --images-dir /workspace/screen2ax_eval/data/screen2ax_linearized_simple/images \\
        --split-info /workspace/data/yolo/split_info.json \\
        --split val \\
        --output-dir /workspace/inference_results_qwen_lora/val
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utils.parser import parse_tree
from inference import clean_model_output

logger = logging.getLogger(__name__)

BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
NON_LORA_FILE = "non_lora_state_dict.bin"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(lora_dir: Path, base_model_id: str, dtype: torch.dtype, attn_impl: str):
    """Load base Qwen3-VL, merge non-LoRA trainable weights, attach LoRA, merge."""
    from transformers import AutoProcessor
    # Qwen3-VL ships a dedicated AutoModelForImageTextToText class path in 5.x transformers.
    from transformers import AutoModelForImageTextToText
    from peft import PeftModel

    logger.info(f"Loading base model: {base_model_id}")
    processor = AutoProcessor.from_pretrained(base_model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    non_lora_path = lora_dir / NON_LORA_FILE
    if non_lora_path.exists():
        logger.info(f"Loading non-LoRA trainables: {non_lora_path}")
        non_lora = torch.load(non_lora_path, map_location="cpu", weights_only=False)
        # Strip prefixes added by PEFT wrapping during training.
        non_lora = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora.items()}
        if any(k.startswith("model.model.") for k in non_lora):
            non_lora = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora.items()}
        missing, unexpected = model.load_state_dict(non_lora, strict=False)
        if unexpected:
            logger.warning(f"Unexpected non-LoRA keys (first 3): {list(unexpected)[:3]}")
    else:
        logger.warning(f"No {NON_LORA_FILE} found in {lora_dir}; skipping merger weights")

    logger.info(f"Attaching LoRA adapter from {lora_dir}")
    model = PeftModel.from_pretrained(model, str(lora_dir))
    logger.info("Merging LoRA weights into base")
    model = model.merge_and_unload()
    model.eval()
    return processor, model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_one(
    processor,
    model,
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    min_pixels: int,
    max_pixels: int,
) -> dict:
    start = time.time()
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image,
                 "min_pixels": min_pixels, "max_pixels": max_pixels},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    prompt_len = int(inputs["input_ids"].shape[-1])

    do_sample = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0, prompt_len:]
    completion_tokens = int(gen_ids.shape[-1])
    raw = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

    latency = time.time() - start
    return {
        "success": True,
        "raw_output": raw,
        "latency": latency,
        "image_width": img_w,
        "image_height": img_h,
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_tokens,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def load_stems(split_info: Path, split: str) -> List[str]:
    data = json.loads(split_info.read_text())
    if split not in ("train", "val"):
        raise SystemExit(f"--split must be train|val, got {split}")
    return list(data[split])


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def run(args: argparse.Namespace) -> None:
    lora_dir = Path(args.lora_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    raw_dir = output_dir / "raw"
    parsed_dir = output_dir / "parsed"
    meta_dir = output_dir / "metadata"
    for d in (raw_dir, parsed_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    stems = load_stems(Path(args.split_info), args.split)
    if args.num_samples is not None:
        stems = stems[: args.num_samples]
    logger.info(f"{args.split}: {len(stems)} samples")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    processor, model = load_model(lora_dir, args.base_model, dtype, args.attn_impl)

    system_prompt = config.SIMPLIFIED_SYSTEM_PROMPT
    user_prompt = config.USER_PROMPT

    n_ok = 0
    n_parse_ok = 0
    n_fail = 0
    total_latency = 0.0
    total_completion = 0

    for stem in tqdm(stems, desc=f"infer[{args.split}]"):
        out_raw = raw_dir / f"{stem}.txt"
        out_parsed = parsed_dir / f"{stem}.txt"
        out_meta = meta_dir / f"{stem}.json"
        if args.resume and out_parsed.exists():
            continue

        image_path = find_image(images_dir, stem)
        if image_path is None:
            logger.warning(f"Image missing for {stem}")
            n_fail += 1
            continue

        try:
            r = generate_one(
                processor, model, image_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
        except Exception as e:
            logger.error(f"{stem}: {e}")
            n_fail += 1
            out_raw.write_text("")
            out_parsed.write_text("")
            out_meta.write_text(json.dumps({
                "filename": image_path.name if image_path else f"{stem}.png",
                "success": False,
                "error": str(e),
            }, indent=2))
            continue

        cleaned = clean_model_output(r["raw_output"], simplified=args.simplified_cleaning)
        nodes = parse_tree(cleaned)
        parse_success = len(nodes) > 0

        out_raw.write_text(r["raw_output"], encoding="utf-8")
        out_parsed.write_text(cleaned, encoding="utf-8")
        out_meta.write_text(json.dumps({
            "filename": image_path.name,
            "model": args.base_model + " + LoRA " + str(lora_dir),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_seconds": r["latency"],
            "image_width": r["image_width"],
            "image_height": r["image_height"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "success": True,
            "parse_success": parse_success,
            "node_count": len(nodes),
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "split": args.split,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        n_ok += 1
        total_latency += r["latency"]
        total_completion += r["completion_tokens"]
        if parse_success:
            n_parse_ok += 1

    total = n_ok + n_fail
    avg_lat = total_latency / n_ok if n_ok else 0.0
    avg_tok = total_completion / n_ok if n_ok else 0.0
    print(f"\n{'='*50}\nInference [{args.split}] done")
    print(f"Processed:       {total}")
    print(f"Success:         {n_ok}  (parse ok: {n_parse_ok})")
    print(f"Failures:        {n_fail}")
    print(f"Avg latency:     {avg_lat:.2f}s")
    print(f"Avg out tokens:  {avg_tok:.0f}")
    print(f"Output:          {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Qwen3-VL-8B LoRA inference locally and save in evaluate.py format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lora-dir", required=True,
                   help="Top-level LoRA training output dir (contains adapter_config.json, non_lora_state_dict.bin)")
    p.add_argument("--base-model", default=BASE_MODEL_ID)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--split-info", required=True,
                   help="split_info.json with {'train': [...], 'val': [...]}")
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--min-pixels", type=int, default=256 * 32 * 32)
    p.add_argument("--max-pixels", type=int, default=1344 * 32 * 32)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--attn-impl", default="sdpa",
                   help="sdpa or flash_attention_2 (if installed)")
    p.add_argument("--simplified-cleaning", action="store_true", default=True,
                   help="Use 7-class cleaning (matches SIMPLIFIED_SYSTEM_PROMPT output)")
    p.add_argument("--resume", action="store_true",
                   help="Skip stems that already have a parsed file")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    run(args)
