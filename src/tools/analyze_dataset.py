"""
Analyze a DetectionDataset JSON to recommend model_max_length,
min_visual_tokens, and max_visual_tokens.

Usage:
    python analyze_dataset.py \
        --data_path  /path/to/metadata/hf_train.json \
        --img_dir    /path/to/images \
        --tokenizer  showlab/ShowUI-2B

The tokenizer is used to count real text tokens.
If --tokenizer is omitted, a rough character-based estimate is used instead.
"""

import argparse
import json
import math
import os
from PIL import Image

PATCH_SIZE = 28  # Qwen2-VL uses 28x28 patches


def visual_tokens_for_image(img_path, min_pixels, max_pixels):
    """
    Replicate the Qwen2-VL dynamic resolution logic:
    resize the image so its total patch count is in [min_pixels, max_pixels],
    keeping the aspect ratio aligned to multiples of PATCH_SIZE.
    Returns the number of visual tokens.
    """
    try:
        img = Image.open(img_path)
        w, h = img.size
    except Exception:
        return None

    total_pixels = w * h
    patch_pixels = PATCH_SIZE * PATCH_SIZE  # 784

    min_p = min_pixels * patch_pixels
    max_p = max_pixels * patch_pixels

    # scale so total pixel count is within [min_p, max_p]
    if total_pixels < min_p:
        scale = math.sqrt(min_p / total_pixels)
    elif total_pixels > max_p:
        scale = math.sqrt(max_p / total_pixels)
    else:
        scale = 1.0

    new_w = max(PATCH_SIZE, round(w * scale / PATCH_SIZE) * PATCH_SIZE)
    new_h = max(PATCH_SIZE, round(h * scale / PATCH_SIZE) * PATCH_SIZE)

    return (new_w // PATCH_SIZE) * (new_h // PATCH_SIZE)


def format_elements_answer(elements):
    lines = []
    for el in elements:
        bbox = el["bbox"]
        bbox_str = "[{}, {}, {}, {}]".format(
            round(bbox[0], 2), round(bbox[1], 2),
            round(bbox[2], 2), round(bbox[3], 2),
        )
        lines.append("{} | {} | {}".format(el["label"], el["type"], bbox_str))
    return "\n".join(lines)


def count_text_tokens(text, tokenizer):
    if tokenizer is None:
        # rough estimate: ~3.5 chars per token for English+symbols
        return max(1, len(text) // 4)
    return len(tokenizer.encode(text, add_special_tokens=False))


def percentile(sorted_list, p):
    idx = int(math.ceil(p / 100.0 * len(sorted_list))) - 1
    return sorted_list[max(0, min(idx, len(sorted_list) - 1))]


def analyze(data_path, img_dir, min_visual_tokens, max_visual_tokens, tokenizer):
    with open(data_path) as f:
        data = json.load(f)

    print(f"\nDataset: {data_path}")
    print(f"Total samples: {len(data)}")
    print(f"Visual token range: [{min_visual_tokens}, {max_visual_tokens}]")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}\n")

    visual_token_counts = []
    text_token_counts   = []
    total_token_counts  = []
    answer_lengths      = []
    n_elements_list     = []
    missing_images      = 0

    # fixed overhead: chat template tokens (approx)
    CHAT_OVERHEAD = 32

    for item in data:
        # --- image ---
        img_path = os.path.join(img_dir, item["img_url"])
        vt = visual_tokens_for_image(img_path, min_visual_tokens, max_visual_tokens)
        if vt is None:
            missing_images += 1
            continue
        visual_token_counts.append(vt)

        # --- text: instruction + system prompt (approx 40 tokens) + answer ---
        instruction_tokens = count_text_tokens(item["instruction"], tokenizer) + 40
        answer = format_elements_answer(item["elements"])
        answer_tokens = count_text_tokens(answer, tokenizer)
        answer_lengths.append(answer_tokens)
        n_elements_list.append(len(item["elements"]))

        text_tokens = instruction_tokens + answer_tokens + CHAT_OVERHEAD
        text_token_counts.append(text_tokens)
        total_token_counts.append(vt + text_tokens)

    if missing_images:
        print(f"  WARNING: {missing_images} images could not be opened and were skipped.\n")

    def stats(name, values):
        values_s = sorted(values)
        print(f"  {name}:")
        print(f"    min={values_s[0]}  "
              f"p50={percentile(values_s,50)}  "
              f"p90={percentile(values_s,90)}  "
              f"p95={percentile(values_s,95)}  "
              f"p99={percentile(values_s,99)}  "
              f"max={values_s[-1]}")

    print("=== Visual tokens (per image after resizing) ===")
    stats("visual_tokens", visual_token_counts)

    print("\n=== Text tokens (instruction + system + answer) ===")
    stats("text_tokens", text_token_counts)

    print("\n=== Answer tokens (elements text only) ===")
    stats("answer_tokens", answer_lengths)

    print("\n=== Number of elements per sample ===")
    stats("n_elements", n_elements_list)

    print("\n=== TOTAL tokens (visual + text) ===")
    stats("total_tokens", total_token_counts)

    # --- Recommendations ---
    total_s = sorted(total_token_counts)
    visual_s = sorted(visual_token_counts)
    p95_total  = percentile(total_s, 95)
    p99_total  = percentile(total_s, 99)
    p5_visual  = percentile(visual_s, 5)
    p95_visual = percentile(visual_s, 95)

    # round up to nearest 256
    def ceil256(x):
        return int(math.ceil(x / 256)) * 256

    rec_max_len     = ceil256(p99_total)
    rec_min_visual  = max(64,  int(math.floor(p5_visual  / 64))  * 64)
    rec_max_visual  = min(max_visual_tokens, ceil256(p95_visual))

    print("\n" + "=" * 55)
    print("RECOMMENDATIONS")
    print("=" * 55)
    print(f"  --model_max_length   {rec_max_len:<8}  (covers p99 of your data)")
    print(f"  --min_visual_tokens  {rec_min_visual:<8}  (p5  of image sizes)")
    print(f"  --max_visual_tokens  {rec_max_visual:<8}  (p95 of image sizes)")
    print()
    print("  Sanity checks:")
    print(f"    p95 total tokens : {p95_total}  (should be < model_max_length)")
    print(f"    p99 total tokens : {p99_total}  (= model_max_length recommendation)")
    print(f"    max total tokens : {total_s[-1]}  (outliers beyond p99 will be skipped by the assert in dset_detection.py)")
    print()
    print("  Note: increase max_visual_tokens for higher resolution at the cost of VRAM.")
    print("        VRAM scales roughly linearly with max_visual_tokens.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Path to hf_train.json")
    parser.add_argument("--img_dir", required=True,
                        help="Path to images/ directory")
    parser.add_argument("--tokenizer", default=None,
                        help="HuggingFace tokenizer id or local path (e.g. showlab/ShowUI-2B). "
                             "If omitted, a character-based estimate is used.")
    parser.add_argument("--min_visual_tokens", type=int, default=256,
                        help="Current min_visual_tokens setting (default 256)")
    parser.add_argument("--max_visual_tokens", type=int, default=1344,
                        help="Current max_visual_tokens setting (default 1344)")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.tokenizer} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    analyze(
        data_path=args.data_path,
        img_dir=args.img_dir,
        min_visual_tokens=args.min_visual_tokens,
        max_visual_tokens=args.max_visual_tokens,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
