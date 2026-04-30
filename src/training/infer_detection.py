"""
Inference script for Screen2AX element detection with fine-tuned ShowUI.

Usage:
    # Run on overfit subset
    python infer_detection.py \
        --checkpoint trained_models/overfit_10/2026-03-22_20-05-06/ckpt_model \
        --dataset_dir datasets/Screen2AX \
        --json hf_overfit \
        --output_dir inference_results

    # Run on a single image
    python infer_detection.py \
        --checkpoint trained_models/overfit_10/2026-03-22_20-05-06/ckpt_model \
        --image path/to/screenshot.png
"""

import os
import re
import sys
import json
import argparse

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.showui.processing_showui import ShowUIProcessor
from model.showui.modeling_showui import ShowUIForConditionalGeneration
from peft import LoraConfig, get_peft_model
from main.eval_screen2ax import compute_iou, match_elements, compute_f1


def find_target_linear_names(model, lora_namespan_exclude=["visual"]):
    """Find linear layers to apply LoRA (must match training config)."""
    linear_cls = torch.nn.modules.Linear
    lora_module_names = []
    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, linear_cls):
            lora_module_names.append(name)
    return lora_module_names


def load_model(checkpoint_dir, model_id="showlab/ShowUI-2B", device="cuda",
               lora_r=32, lora_alpha=64):
    """Load ShowUI model with LoRA weights from DeepSpeed checkpoint.

    Steps:
    1. Load base model
    2. Create PEFT/LoRA model (same config as training)
    3. Convert DeepSpeed ZeRO checkpoint to fp32
    4. Load state dict into the PEFT model
    5. Merge LoRA weights into base model for faster inference
    """
    # Read training args if available
    exp_dir = os.path.dirname(checkpoint_dir)  # go up from ckpt_model/
    args_path = os.path.join(exp_dir, "args.json")
    if os.path.exists(args_path):
        with open(args_path) as f:
            train_args = json.load(f)
        lora_r = train_args.get("lora_r", lora_r)
        lora_alpha = train_args.get("lora_alpha", lora_alpha)
        print(f"Loaded training args: lora_r={lora_r}, lora_alpha={lora_alpha}")

    print(f"Loading base model: {model_id}")
    processor = ShowUIProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1344 * 28 * 28,
        model_max_length=8192,
    )

    model = ShowUIForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # load on CPU first for weight merging
    )

    if checkpoint_dir and lora_r > 0:
        # Step 2: Create PEFT model with same LoRA config as training
        print(f"Creating LoRA model (r={lora_r}, alpha={lora_alpha})...")
        lora_target_modules = find_target_linear_names(model, lora_namespan_exclude=["visual"])
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Step 3: Convert DeepSpeed ZeRO checkpoint to fp32 state dict
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        print(f"Converting DeepSpeed checkpoint: {checkpoint_dir}")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)

        # Step 4: Load into PEFT model
        lora_keys = [k for k in state_dict if "lora_" in k]
        print(f"  Found {len(lora_keys)} LoRA parameters in checkpoint")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        # Step 5: Merge LoRA into base model
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    model = model.to(device).to(torch.bfloat16)
    model.eval()
    return model, processor


def run_inference(model, processor, image, max_new_tokens=2048, device="cuda"):
    """Run element detection inference on a single image."""

    system_prompt = (
        "Detect all interactive UI elements in this macOS screenshot. "
        "For each element, output its accessibility type and bounding box coordinates. "
        "The bounding box coordinates [x1, y1, x2, y2] are relative to the screenshot, "
        "scaled from 0 to 1000. Output format: Type [x1, y1, x2, y2]; Type [x1, y1, x2, y2]; ..."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {
                    "type": "image",
                    "min_pixels": processor.image_processor.min_pixels,
                    "max_pixels": processor.image_processor.max_pixels,
                },
            ],
        }
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Rename for model.generate
    if "image_grid_thw" in inputs:
        pass  # generate handles this via prepare_inputs_for_generation

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    # Strip input tokens from output
    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    return output_text


def parse_elements(output_text):
    """Parse model output into list of (type, x1, y1, x2, y2) tuples."""
    pattern = r'(\w+)\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(pattern, output_text)
    elements = []
    for role, x1, y1, x2, y2 in matches:
        elements.append({
            "type": role,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_norm": [int(x1) / 1000, int(y1) / 1000, int(x2) / 1000, int(y2) / 1000],
        })
    return elements


CLASS_COLORS = [
    "red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow",
    "lime", "deeppink", "dodgerblue", "gold", "tomato", "springgreen",
    "orchid", "turquoise", "salmon", "chartreuse", "mediumpurple", "coral",
]
_class_color_map = {}


def get_class_color(class_name):
    """Return a consistent color for each class name."""
    if class_name not in _class_color_map:
        _class_color_map[class_name] = CLASS_COLORS[len(_class_color_map) % len(CLASS_COLORS)]
    return _class_color_map[class_name]


def draw_elements(image, elements, output_path=None):
    """Draw bounding boxes on the image with consistent class colors."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for elem in elements:
        color = get_class_color(elem["type"])
        x1 = elem["bbox_norm"][0] * w
        y1 = elem["bbox_norm"][1] * h
        x2 = elem["bbox_norm"][2] * w
        y2 = elem["bbox_norm"][3] * h
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), elem["type"], fill=color)

    if output_path:
        img.save(output_path)

    return img


def run_on_dataset(model, processor, dataset_dir, json_name, output_dir, max_new_tokens=2048):
    """Run inference on a dataset and compare with ground truth."""
    img_dir = os.path.join(dataset_dir, "images")
    meta_dir = os.path.join(dataset_dir, "metadata")

    with open(os.path.join(meta_dir, f"{json_name}.json")) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    results = []
    results_path = os.path.join(output_dir, "results.json")

    for i, item in enumerate(data):
        img_path = os.path.join(img_dir, item["img_url"])
        image = Image.open(img_path).convert("RGB")

        print(f"\n[{i+1}/{len(data)}] {item['img_url']} (GT: {item['element_size']} elements)")

        output_text = run_inference(model, processor, image, max_new_tokens)
        pred_elements = parse_elements(output_text)

        # Parse GT elements for visualization
        gt_elements = parse_elements(item.get("all_elements_str", ""))

        # Compute F1@IoU=0.1
        pred_tuples = [(e["type"], e["bbox"]) for e in pred_elements]
        gt_tuples = [(e["type"], e["bbox"]) for e in gt_elements]
        tp, fp, fn = match_elements(pred_tuples, gt_tuples, iou_threshold=0.1)
        precision, recall, f1 = compute_f1(tp, fp, fn)

        print(f"  Predicted: {len(pred_elements)} elements")
        print(f"  F1@IoU=0.1: {f1:.4f}  (P={precision:.4f} R={recall:.4f} TP={tp} FP={fp} FN={fn})")
        print(f"  Output: {output_text[:200]}...")

        # Draw GT and predictions
        gt_vis_path = os.path.join(vis_dir, f"{item['img_url'].replace('.png', '')}_gt.png")
        draw_elements(image, gt_elements, gt_vis_path)

        pred_vis_path = os.path.join(vis_dir, f"{item['img_url'].replace('.png', '')}_pred.png")
        draw_elements(image, pred_elements, pred_vis_path)

        result = {
            "img_url": item["img_url"],
            "gt_elements": item["element_size"],
            "pred_elements": len(pred_elements),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1_iou01": f1,
            "gt_answer": item.get("all_elements_str", ""),
            "pred_answer": output_text,
            "pred_parsed": pred_elements,
        }
        results.append(result)

        # Write results after each sample
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Print summary
    total_gt = sum(r["gt_elements"] for r in results)
    total_pred = sum(r["pred_elements"] for r in results)
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    agg_precision, agg_recall, agg_f1 = compute_f1(total_tp, total_fp, total_fn)
    avg_f1 = sum(r["f1_iou01"] for r in results) / len(results)

    print(f"\nSummary: {len(results)} images")
    print(f"  GT elements total: {total_gt}")
    print(f"  Predicted elements total: {total_pred}")
    print(f"  Aggregate F1@IoU=0.1: {agg_f1:.4f}  (P={agg_precision:.4f} R={agg_recall:.4f})")
    print(f"  Mean per-image F1:     {avg_f1:.4f}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ShowUI Element Detection Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ckpt_model dir")
    parser.add_argument("--model_id", type=str, default="showlab/ShowUI-2B")
    parser.add_argument("--dataset_dir", type=str, help="Path to Screen2AX dataset dir")
    parser.add_argument("--json", type=str, default="hf_overfit", help="JSON split name")
    parser.add_argument("--image", type=str, help="Single image path for inference")
    parser.add_argument("--output_dir", type=str, default="inference_results")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, processor = load_model(args.checkpoint, args.model_id, args.device)

    if args.image:
        # Single image inference
        image = Image.open(args.image).convert("RGB")
        output_text = run_inference(model, processor, image, args.max_new_tokens, args.device)
        elements = parse_elements(output_text)
        print(f"\nRaw output:\n{output_text}")
        print(f"\nParsed {len(elements)} elements:")
        for e in elements:
            print(f"  {e['type']} {e['bbox']}")

        os.makedirs(args.output_dir, exist_ok=True)
        vis_path = os.path.join(args.output_dir, "prediction.png")
        draw_elements(image, elements, vis_path)

    elif args.dataset_dir:
        # Dataset inference
        run_on_dataset(
            model, processor, args.dataset_dir, args.json,
            args.output_dir, args.max_new_tokens
        )
    else:
        parser.error("Provide either --dataset_dir or --image")


if __name__ == "__main__":
    main()
