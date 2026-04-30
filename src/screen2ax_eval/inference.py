#!/usr/bin/env python3
"""Run VLM inference on Screen2AX test set via vLLM OpenAI-compatible API."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# Ensure package imports work when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from utils.parser import parse_tree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt cleaning
# ---------------------------------------------------------------------------

# Pattern to match the start of a valid AX tree line
AX_ROLE_PATTERN = re.compile(
    r"^\s*(?:" + "|".join(re.escape(r) for r in config.VALID_ROLES) + r")"
    r"(?:\([^)]*\))?\s*\["
)

# Also match roles without AX prefix (model might output either)
AX_ROLE_PATTERN_NO_PREFIX = re.compile(
    r"^\s*(?:" + "|".join(re.escape(r) for r in config.VALID_ROLES_NO_PREFIX) + r")"
    r"(?:\([^)]*\))?\s*\["
)

# Simplified 7-class patterns
SIMPLIFIED_ROLE_PATTERN = re.compile(
    r"^\s*(?:" + "|".join(re.escape(r) for r in config.SIMPLIFIED_ROLES) + r")"
    r"(?:\([^)]*\))?\s*\["
)
_simplified_no_prefix = {r[2:] if r.startswith("AX") else r for r in config.SIMPLIFIED_ROLES}
SIMPLIFIED_ROLE_PATTERN_NO_PREFIX = re.compile(
    r"^\s*(?:" + "|".join(re.escape(r) for r in _simplified_no_prefix) + r")"
    r"(?:\([^)]*\))?\s*\["
)


def clean_model_output(text: str, simplified: bool = False) -> str:
    """Clean VLM output: strip markdown fences, commentary, etc.

    Returns the cleaned text containing only the AX tree lines.
    When *simplified* is True, only lines matching the 7-class roles are kept.
    """
    if not text or not text.strip():
        return ""

    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```\w*\n?", "", text)
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    text = text.strip()

    lines = text.split("\n")

    if simplified:
        pat = SIMPLIFIED_ROLE_PATTERN
        pat_np = SIMPLIFIED_ROLE_PATTERN_NO_PREFIX
    else:
        pat = AX_ROLE_PATTERN
        pat_np = AX_ROLE_PATTERN_NO_PREFIX

    # Find first and last lines that look like AX tree entries
    first_idx = None
    last_idx = None
    for i, line in enumerate(lines):
        if pat.match(line) or pat_np.match(line):
            if first_idx is None:
                first_idx = i
            last_idx = i

    if first_idx is None:
        # No valid AX lines found -- might be a refusal or empty response
        return ""

    return "\n".join(lines[first_idx : last_idx + 1])


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_base64(image_path: Path) -> str:
    """Load image and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Return (width, height) of an image."""
    with Image.open(image_path) as img:
        return img.size


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------

def infer_single(
    client: OpenAI,
    image_path: Path,
    model: str,
    max_tokens: int,
    temperature: float,
    simplified: bool = False,
) -> dict:
    """Run inference on a single image. Returns dict with prediction and metadata."""
    start = time.time()
    img_b64 = load_image_base64(image_path)
    img_w, img_h = get_image_dimensions(image_path)

    system_prompt = config.SIMPLIFIED_SYSTEM_PROMPT if simplified else config.SYSTEM_PROMPT

    # Determine mime type
    suffix = image_path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_map.get(suffix, "image/png")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{img_b64}",
                            },
                        },
                        {"type": "text", "text": config.USER_PROMPT},
                    ],
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        raw_output = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

    except Exception as e:
        latency = time.time() - start
        logger.error(f"API error for {image_path.name}: {e}")
        return {
            "success": False,
            "error": str(e),
            "raw_output": "",
            "cleaned_output": "",
            "latency": latency,
            "image_width": img_w,
            "image_height": img_h,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    latency = time.time() - start
    cleaned = clean_model_output(raw_output, simplified=simplified)

    # Validate: try to parse
    nodes = parse_tree(cleaned)
    parse_success = len(nodes) > 0

    return {
        "success": True,
        "parse_success": parse_success,
        "raw_output": raw_output,
        "cleaned_output": cleaned,
        "latency": latency,
        "image_width": img_w,
        "image_height": img_h,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "node_count": len(nodes),
    }


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def infer_with_retry(
    client: OpenAI,
    image_path: Path,
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int = 3,
    simplified: bool = False,
) -> dict:
    """Run inference with exponential backoff retries."""
    for attempt in range(max_retries):
        result = infer_single(client, image_path, model, max_tokens, temperature, simplified=simplified)
        if result["success"]:
            return result
        if attempt < max_retries - 1:
            wait = 2 ** attempt
            logger.warning(
                f"Retry {attempt + 1}/{max_retries} for {image_path.name} "
                f"after {wait}s: {result.get('error', 'unknown')}"
            )
            time.sleep(wait)
    return result


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:
    """Run inference on the dataset."""
    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / "images"
    output_dir = Path(args.output_dir)

    # Create output directories
    raw_dir = output_dir / "raw"
    parsed_dir = output_dir / "parsed"
    meta_dir = output_dir / "metadata"
    for d in [raw_dir, parsed_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Discover images
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]
    )
    if not image_files:
        logger.error(f"No images found in {images_dir}")
        return

    logger.info(f"Found {len(image_files)} images in {images_dir}")

    # Limit samples if requested
    if args.num_samples is not None:
        image_files = image_files[: args.num_samples]
        logger.info(f"Limiting to {len(image_files)} samples")

    # Filter already processed if --resume
    if args.resume:
        remaining = []
        for img in image_files:
            pred_path = parsed_dir / (img.stem + ".txt")
            if not pred_path.exists():
                remaining.append(img)
        skipped = len(image_files) - len(remaining)
        logger.info(f"Resuming: skipping {skipped} already processed, {len(remaining)} remaining")
        image_files = remaining

    if not image_files:
        logger.info("All samples already processed.")
        return

    # Initialize client
    client = OpenAI(base_url=args.base_url, api_key=args.api_key or "dummy")

    # Stats tracking
    total_latency = 0.0
    total_tokens = 0
    n_success = 0
    n_parse_success = 0
    n_failures = 0

    def process_one(image_path: Path) -> dict:
        """Process a single image and save results."""
        result = infer_with_retry(
            client, image_path, args.model, args.max_tokens, args.temperature,
            simplified=args.simplified_roles,
        )

        stem = image_path.stem

        # Save raw output
        with open(raw_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write(result.get("raw_output", ""))

        # Save cleaned/parsed output
        with open(parsed_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write(result.get("cleaned_output", ""))

        # Save metadata
        metadata = {
            "filename": image_path.name,
            "model": args.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_seconds": result["latency"],
            "image_width": result["image_width"],
            "image_height": result["image_height"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "success": result["success"],
            "parse_success": result.get("parse_success", False),
            "node_count": result.get("node_count", 0),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "system_prompt": config.SYSTEM_PROMPT,
            "user_prompt": config.USER_PROMPT,
        }
        if not result["success"]:
            metadata["error"] = result.get("error", "")

        with open(meta_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return result

    if args.batch_size > 1:
        # Concurrent inference
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            futures = {
                executor.submit(process_one, img): img for img in image_files
            }
            with tqdm(total=len(image_files), desc="Inference") as pbar:
                for future in as_completed(futures):
                    img = futures[future]
                    try:
                        result = future.result()
                        if result["success"]:
                            n_success += 1
                            total_latency += result["latency"]
                            total_tokens += result["completion_tokens"]
                            if result.get("parse_success"):
                                n_parse_success += 1
                        else:
                            n_failures += 1
                    except Exception as e:
                        logger.error(f"Unexpected error for {img.name}: {e}")
                        n_failures += 1
                    pbar.update(1)
    else:
        # Sequential inference
        for image_path in tqdm(image_files, desc="Inference"):
            try:
                result = process_one(image_path)
                if result["success"]:
                    n_success += 1
                    total_latency += result["latency"]
                    total_tokens += result["completion_tokens"]
                    if result.get("parse_success"):
                        n_parse_success += 1
                else:
                    n_failures += 1
            except Exception as e:
                logger.error(f"Unexpected error for {image_path.name}: {e}")
                n_failures += 1

    # Summary
    total = n_success + n_failures
    avg_latency = total_latency / n_success if n_success else 0
    avg_tokens = total_tokens / n_success if n_success else 0

    print(f"\n{'='*50}")
    print(f"Inference Summary")
    print(f"{'='*50}")
    print(f"Model:              {args.model}")
    print(f"Total processed:    {total}")
    print(f"Successful:         {n_success}")
    print(f"Parse success:      {n_parse_success}")
    print(f"Failures:           {n_failures}")
    print(f"Avg latency:        {avg_latency:.2f}s")
    print(f"Avg output tokens:  {avg_tokens:.0f}")
    print(f"Results saved to:   {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM inference on Screen2AX test set via vLLM OpenAI-compatible API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=config.DATASET_DIR,
        help="Path to dataset with images/ and annotations/ subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(config.RESULTS_DIR, "default_run"),
        help="Where to save predictions (raw/, parsed/, metadata/)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=config.VLLM_BASE_URL,
        help="vLLM server base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_NAME,
        help="Model name for the API",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "dummy"),
        help="API key (default: OPENAI_API_KEY env var or 'dummy' for local vLLM)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of samples (for testing). Omit to run all.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=config.DEFAULT_MAX_TOKENS,
        help="Max output tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.DEFAULT_TEMPERATURE,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images that already have a prediction file in output-dir",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of concurrent requests (increase for throughput)",
    )
    parser.add_argument(
        "--simplified-roles",
        action="store_true",
        help="Use 7-class simplified role mapping (simplified prompt + cleaning)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    run_inference(args)
