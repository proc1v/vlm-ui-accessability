"""
Inference script for Screen2AX-Tree accessibility tree generation with fine-tuned ShowUI.

Usage:
    # Run on val split
    python infer_tree.py \
        --checkpoint trained_models/screen2ax_tree/2026-04-10_12-00-00/ckpt_model \
        --dataset_dir datasets/Screen2AX_linearized \
        --split val \
        --output_dir inference_results_tree

    # Run on a single image
    python infer_tree.py \
        --checkpoint trained_models/screen2ax_tree/2026-04-10_12-00-00/ckpt_model \
        --image path/to/screenshot.png \
        --output_dir inference_results_tree
"""

import os
import sys
import json
import argparse

import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from peft import LoraConfig, get_peft_model

SHOWUI_MODELS = ["showlab/ShowUI-2B"]
QWEN2_VL_MODELS = ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"]
QWEN2_5_VL_MODELS = ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]

CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
    "{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}{% endif %}"
    "{% endfor %}<|im_end|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

HUMAN_PROMPT = (
    "Generate the complete accessibility tree for this UI screenshot. "
    "Output each element on a new line with indentation showing hierarchy. "
    "Format: ROLE [x,y,w,h] attributes"
)


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


def _load_processor(model_id, min_pixels, max_pixels):
    if model_id in SHOWUI_MODELS:
        from model.showui.processing_showui import ShowUIProcessor
        proc = ShowUIProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels, model_max_length=8192,
        )
    elif model_id in QWEN2_VL_MODELS:
        from model.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        proc = Qwen2VLProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels, model_max_length=8192,
        )
    elif model_id in QWEN2_5_VL_MODELS:
        from model.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        proc = Qwen2_5_VLProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels, model_max_length=8192,
        )
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
    proc.chat_template = CHAT_TEMPLATE
    proc.tokenizer.chat_template = CHAT_TEMPLATE
    return proc


def _load_base_model(model_id):
    if model_id in SHOWUI_MODELS:
        from model.showui.modeling_showui import ShowUIForConditionalGeneration
        return ShowUIForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cpu",
        )
    elif model_id in QWEN2_VL_MODELS:
        from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cpu",
        )
    elif model_id in QWEN2_5_VL_MODELS:
        from model.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cpu",
        )
    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def load_model(checkpoint_dir, model_id="Qwen/Qwen2-VL-7B-Instruct", device="cuda",
               lora_r=32, lora_alpha=64):
    """Load model with LoRA weights from DeepSpeed checkpoint."""
    exp_dir = os.path.dirname(checkpoint_dir)
    args_path = os.path.join(exp_dir, "args.json")
    if os.path.exists(args_path):
        with open(args_path) as f:
            train_args = json.load(f)
        lora_r = train_args.get("lora_r", lora_r)
        lora_alpha = train_args.get("lora_alpha", lora_alpha)
        model_id = train_args.get("model_id", model_id)
        print(f"Loaded training args: model_id={model_id}, lora_r={lora_r}, lora_alpha={lora_alpha}")

    print(f"Loading base model: {model_id}")
    processor = _load_processor(model_id, min_pixels=256 * 28 * 28, max_pixels=1344 * 28 * 28)
    model = _load_base_model(model_id)

    if checkpoint_dir and lora_r > 0:
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

        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        print(f"Converting DeepSpeed checkpoint: {checkpoint_dir}")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)

        lora_keys = [k for k in state_dict if "lora_" in k]
        print(f"  Found {len(lora_keys)} LoRA parameters in checkpoint")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    model = model.to(device).to(torch.bfloat16)
    model.eval()
    return model, processor


def run_inference(model, processor, image, max_new_tokens=4096, device="cuda"):
    """Run accessibility tree generation inference on a single image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": HUMAN_PROMPT},
                {"type": "image"},
            ],
        }
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    return output_text


def run_on_dataset(model, processor, dataset_dir, split, output_dir, max_new_tokens=4096, device="cuda"):
    """Run inference on a dataset split and save predictions to JSON."""
    json_path = os.path.join(dataset_dir, f"{split}.json")
    with open(json_path) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"predictions_{split}.json")

    results = []

    for i, item in enumerate(data):
        img_path = os.path.join(dataset_dir, item["image"])
        image = Image.open(img_path).convert("RGB")

        gt_text = item["conversations"][1]["value"]
        print(f"\n[{i+1}/{len(data)}] {item['image']}")

        pred_text = run_inference(model, processor, image, max_new_tokens, device)

        result = {
            "id": item["id"],
            "image": item["image"],
            "gt": gt_text,
            "pred": pred_text,
        }
        results.append(result)

        # Write incrementally
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  GT lines:   {len(gt_text.splitlines())}")
        print(f"  Pred lines: {len(pred_text.splitlines())}")
        print(f"  Pred (first 200 chars): {pred_text[:200]}")

    print(f"\nDone. {len(results)} predictions saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="ShowUI Accessibility Tree Generation Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ckpt_model dir")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--dataset_dir", type=str, help="Path to Screen2AX_linearized dataset dir")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (train/val)")
    parser.add_argument("--image", type=str, help="Single image path for inference")
    parser.add_argument("--output_dir", type=str, default="inference_results_tree")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, processor = load_model(args.checkpoint, args.model_id, args.device)

    if args.image:
        image = Image.open(args.image).convert("RGB")
        output_text = run_inference(model, processor, image, args.max_new_tokens, args.device)
        print(f"\nRaw output:\n{output_text}")

        os.makedirs(args.output_dir, exist_ok=True)
        result = {"image": args.image, "pred": output_text}
        out_path = os.path.join(args.output_dir, "prediction.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_path}")

    elif args.dataset_dir:
        run_on_dataset(
            model, processor, args.dataset_dir, args.split,
            args.output_dir, args.max_new_tokens, args.device,
        )
    else:
        parser.error("Provide either --dataset_dir or --image")


if __name__ == "__main__":
    main()
