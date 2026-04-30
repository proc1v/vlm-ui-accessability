"""
Screen2AX → ShowUI: Single-Shot Element Detection

This module provides:
1. convert_screen2ax()     — converts HuggingFace Screen2AX-Element to ShowUI-compatible JSON + images
2. ElementDetectionDataset — custom PyTorch Dataset that plugs into ShowUI's training pipeline
3. Template/prompt design  — for single-shot "detect all elements" task

=== TASK FORMULATION ===

Instead of ShowUI's default grounding:
    User:  "Find the search button"  →  Model: [0.45, 0.07]

We do single-shot detection:
    User:  <image> "Detect all interactive UI elements with their types and bounding boxes."
    Model: "AXButton [54, 12, 82, 44]; AXButton [94, 12, 122, 44]; AXTextArea [13, 69, 345, 107]"

Coordinates in the model output use the [0, 1000] integer scale (same as xy_int=True in ShowUI)
because it produces shorter, cleaner tokens than floats.

=== USAGE ===

Step 1: Convert data
    python screen2ax_showui.py --convert --output_dir ./Screen2AX

Step 2: Use in ShowUI training
    Copy this file into ShowUI's data/ directory, register in dataset.py,
    and point training to your dataset.
"""

import os
import json
import random
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


# ============================================================================
# PART 1: DATA CONVERSION — Screen2AX-Element → ShowUI JSON + images
# ============================================================================

def normalize_bbox(bbox, img_width, img_height):
    """Pixel bbox [x1, y1, x2, y2] → normalized [0, 1]."""
    x1, y1, x2, y2 = bbox
    return [
        x1 / img_width,
        y1 / img_height,
        x2 / img_width,
        y2 / img_height,
    ]


def bbox_to_center(norm_bbox):
    """Normalized bbox → center point."""
    x1, y1, x2, y2 = norm_bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def convert_screen2ax(output_dir="./Screen2AX", min_elements=1, splits=None):
    """
    Download Screen2AX-Element from HuggingFace and convert to ShowUI format.

    Creates:
        output_dir/
            images/       — PNG screenshots
            metadata/
                hf_train.json
                hf_val.json
                hf_test.json

    Each JSON entry has the STANDARD ShowUI grounding format (for compatibility)
    PLUS an extra "all_elements" field used by ElementDetectionDataset.

    JSON entry format:
    {
        "img_url": "train_00001.png",
        "img_size": [1920, 1080],
        "element": [                          # standard ShowUI format (kept for compat)
            {
                "instruction": "AXButton",
                "bbox": [0.028, 0.011, 0.043, 0.041],
                "point": [0.035, 0.026]
            }, ...
        ],
        "element_size": 17,
        "all_elements_str": "AXButton [28, 11, 43, 41]; AXButton [49, 11, 64, 41]; ..."
            ^ pre-formatted answer string for single-shot detection (coords in 0-1000 scale)
    }
    """
    from datasets import load_dataset

    if splits is None:
        splits = ["train", "valid", "test"]

    images_dir = os.path.join(output_dir, "images")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    print("Loading Screen2AX-Element from HuggingFace...")
    dataset = load_dataset("macpaw-research/Screen2AX-Element")

    # Screen2AX uses "valid", ShowUI uses "val"
    split_name_map = {"train": "train", "valid": "val", "test": "test"}

    for split in splits:
        if split not in dataset:
            print(f"  Warning: split '{split}' not found, skipping")
            continue

        showui_name = split_name_map.get(split, split)
        data_list = []
        total_elements = 0

        for idx, sample in enumerate(dataset[split]):
            image = sample["image"]
            objects = sample["objects"]
            bboxes = objects["bbox"]
            categories = objects["category"]

            img_w, img_h = image.size

            element_list = []
            all_elements_parts = []

            for bbox_px, category in zip(bboxes, categories):
                x1, y1, x2, y2 = bbox_px

                # Clamp to image bounds
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))

                # Skip degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                norm = normalize_bbox([x1, y1, x2, y2], img_w, img_h)
                center = bbox_to_center(norm)

                # Standard ShowUI element entry
                element_list.append({
                    "instruction": category,
                    "bbox": norm,
                    "point": center,
                })

                # Pre-format the 0-1000 integer bbox for the detection answer
                bbox_int = [int(v * 1000) for v in norm]
                all_elements_parts.append(
                    f"{category} [{bbox_int[0]}, {bbox_int[1]}, {bbox_int[2]}, {bbox_int[3]}]"
                )

            if len(element_list) < min_elements:
                continue

            # Save image
            img_filename = f"{showui_name}_{idx:05d}.png"
            image.save(os.path.join(images_dir, img_filename))

            # Build the full answer string
            all_elements_str = "; ".join(all_elements_parts)

            entry = {
                "img_url": img_filename,
                "img_size": [img_w, img_h],
                "element": element_list,
                "element_size": len(element_list),
                "all_elements_str": all_elements_str,
            }
            data_list.append(entry)
            total_elements += len(element_list)

        json_path = os.path.join(metadata_dir, f"hf_{showui_name}.json")
        with open(json_path, "w") as f:
            json.dump(data_list, f, indent=2)

        print(f"  [{showui_name}] {len(data_list)} images, {total_elements} elements → {json_path}")

    print("\nDone! Directory structure:")
    print(f"  {output_dir}/images/   — screenshot PNGs")
    print(f"  {output_dir}/metadata/ — JSON metadata files")


# ============================================================================
# PART 2: PROMPT TEMPLATES — for single-shot element detection
# ============================================================================

# System prompts (diversity helps regularize training)
_SYSTEM_DETECT = [
    "Detect all interactive UI elements in this macOS screenshot. For each element, output its accessibility type and bounding box coordinates.",
    "Identify every interactive element visible in this screenshot. Report each element's type and bounding box.",
    "Analyze this macOS screenshot and list all UI elements with their types and locations.",
    "Find all interactive components in this interface screenshot. Output each element's accessibility role and bounding box.",
    "Scan this macOS application screenshot and detect all interactive UI elements with their types and positions.",
    "List every interactive element in this screenshot with its accessibility type and bounding box coordinates.",
    "Examine this macOS interface and identify all clickable, editable, and interactive elements with their bounding boxes.",
    "Detect and catalog all UI elements present in this macOS screenshot, providing type and location for each.",
]

_SYSTEM_FORMAT = (
    " The bounding box coordinates [x1, y1, x2, y2] are relative to the screenshot, "
    "scaled from 0 to 1000. Output format: Type [x1, y1, x2, y2]; Type [x1, y1, x2, y2]; ..."
)


def detection_to_qwen(image_dict, shuffle_image_token=False):
    """
    Build the Qwen2-VL chat message for single-shot detection.

    Returns a list of message dicts ready for processor.apply_chat_template().

    The conversation structure:
        User: {system_prompt} <image>
        Assistant: AXButton [28, 11, 43, 41]; AXButton [49, 11, 64, 41]; ...
    """
    system_prompt = random.choice(_SYSTEM_DETECT) + _SYSTEM_FORMAT

    user_content = []

    if shuffle_image_token:
        # Randomly place image before or after the text
        if random.random() < 0.5:
            user_content.append(image_dict)
            user_content.append({"type": "text", "text": system_prompt})
        else:
            user_content.append({"type": "text", "text": system_prompt})
            user_content.append(image_dict)
    else:
        # Default: system text first, then image
        user_content.append({"type": "text", "text": system_prompt})
        user_content.append(image_dict)

    return [{"role": "user", "content": user_content}]


# ============================================================================
# PART 3: CUSTOM DATASET CLASS — plugs into ShowUI's training loop
# ============================================================================

class ElementDetectionDataset(Dataset):
    """
    Single-shot element detection dataset for ShowUI fine-tuning.

    Input:  screenshot + "Detect all interactive UI elements..."
    Output: "AXButton [28, 11, 43, 41]; AXTextArea [13, 69, 345, 107]; ..."

    This class mirrors ShowUI's GroundingDataset interface so it can be used
    as a drop-in replacement in the training pipeline.

    Key differences from GroundingDataset:
    - Outputs ALL elements at once (not one per query)
    - Answer is a semicolon-separated string (not a single coordinate)
    - Uses custom detection prompts instead of grounding prompts
    """

    def __init__(
        self,
        dataset_dir,
        dataset,          # e.g., "screen2ax"
        json_data,        # e.g., "hf_train"
        processor,
        inference=False,
        args_dict=None,
    ):
        if args_dict is None:
            args_dict = {}

        self.processor = processor
        self.min_pixels = processor.image_processor.min_pixels
        self.max_pixels = processor.image_processor.max_pixels
        self.inference = inference

        # Dataset mapping — add your dataset name here
        dataset_dir_mapping = {
            "screen2ax": "Screen2AX",
        }
        dataset_folder = dataset_dir_mapping.get(dataset, dataset)

        self.base_dir = os.path.join(dataset_dir, dataset_folder)
        self.img_dir = os.path.join(self.base_dir, "images")
        meta_dir = os.path.join(self.base_dir, "metadata")

        with open(os.path.join(meta_dir, f"{json_data}.json")) as f:
            self.json_data = json.load(f)

        self.samples_per_epoch = args_dict.get("samples_per_epoch", len(self.json_data))
        self.random_sample = args_dict.get("random_sample", False)
        self.shuffle_image_token = args_dict.get("shuffle_image_token", False)
        self.max_elements = args_dict.get("max_elements", None)  # optional cap

        print(f"ElementDetectionDataset: {dataset}/{json_data} — {len(self.json_data)} samples")

    def __len__(self):
        if self.random_sample:
            return self.samples_per_epoch
        return len(self.json_data)

    def __getitem__(self, idx):
        while True:
            try:
                return self.get_sample(idx)
            except Exception as e:
                print(f"Error at idx {idx}: {e}")
                idx = random.randint(0, len(self.json_data) - 1)

    def get_sample(self, idx):
        if self.random_sample:
            idx = random.randint(0, len(self.json_data) - 1)
        idx = idx % len(self.json_data)

        item = self.json_data[idx]

        # Load image
        image_path = os.path.join(self.img_dir, item["img_url"])
        image = Image.open(image_path).convert("RGB")
        image_list = [image]

        # Build the answer string
        # Use pre-formatted string if available, otherwise build from elements
        if "all_elements_str" in item:
            answer_str = item["all_elements_str"]
        else:
            # Build from element list
            parts = []
            for elem in item["element"]:
                bbox_int = [int(v * 1000) for v in elem["bbox"]]
                parts.append(
                    f"{elem['instruction']} [{bbox_int[0]}, {bbox_int[1]}, {bbox_int[2]}, {bbox_int[3]}]"
                )
            answer_str = "; ".join(parts)

        # Optionally cap the number of elements (for memory/sequence length)
        if self.max_elements and item["element_size"] > self.max_elements:
            elements = item["element"][: self.max_elements]
            parts = []
            for elem in elements:
                bbox_int = [int(v * 1000) for v in elem["bbox"]]
                parts.append(
                    f"{elem['instruction']} [{bbox_int[0]}, {bbox_int[1]}, {bbox_int[2]}, {bbox_int[3]}]"
                )
            answer_str = "; ".join(parts)

        # Build conversation
        img_dict = {
            "type": "image",
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        source = detection_to_qwen(img_dict, self.shuffle_image_token)

        # Tokenize the prompt (user turn)
        prompt = self.processor.tokenizer.apply_chat_template(
            source, tokenize=False, add_generation_prompt=True
        )
        data_dict_q = self.processor(
            text=prompt,
            images=image_list,
            return_tensors="pt",
            training=not self.inference,
        )

        # Append answer tokens and build labels
        # This replicates ShowUI's batch_add_answer logic but for a string answer
        data_dict_qa = self._add_string_answer(data_dict_q, answer_str)

        max_seq_len = self.processor.tokenizer.model_max_length

        data_dict = dict(
            input_ids=data_dict_qa["input_ids"][0],
            image_sizes=data_dict_qa["image_grid_thw"],
            pixel_values=data_dict_qa["pixel_values"],
            labels=data_dict_qa["labels"][0],
        )

        # Sequence length check
        seq_len = data_dict["input_ids"].shape[0]
        if seq_len > max_seq_len:
            print(f"  Warning: seq len {seq_len} > max {max_seq_len} for {item['img_url']}")

        # Pass through ShowUI-specific keys if present (UI-guided token selection)
        for key in ["select_mask", "patch_pos", "patch_assign", "patch_assign_len"]:
            if key in data_dict_q:
                data_dict[key] = data_dict_q[key]

        return data_dict, item

    def _add_string_answer(self, data_dict_q, answer_str):
        """
        Append the answer string to the tokenized prompt and create labels.

        Labels are set to -100 for the prompt tokens (we don't compute loss on them)
        and to the actual token IDs for the answer tokens.
        """
        tokenizer = self.processor.tokenizer

        # Tokenize the answer
        answer_tokens = tokenizer(
            answer_str,
            return_tensors="pt",
            add_special_tokens=False,
        )
        answer_ids = answer_tokens["input_ids"][0]

        # Add EOS token
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            # Qwen2-VL uses <|im_end|> as EOS in chat
            eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_tensor = torch.tensor([eos_id], dtype=answer_ids.dtype)
        answer_ids = torch.cat([answer_ids, eos_tensor])

        # Get prompt input IDs
        prompt_ids = data_dict_q["input_ids"][0]
        prompt_len = prompt_ids.shape[0]

        # Concatenate: [prompt_tokens] + [answer_tokens] + [eos]
        full_ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)

        # Labels: -100 for prompt (no loss), actual IDs for answer
        labels = torch.cat([
            torch.full((prompt_len,), -100, dtype=full_ids.dtype),
            answer_ids,
        ]).unsqueeze(0)

        # Update attention mask
        if "attention_mask" in data_dict_q:
            prompt_mask = data_dict_q["attention_mask"][0]
            answer_mask = torch.ones(answer_ids.shape[0], dtype=prompt_mask.dtype)
            full_mask = torch.cat([prompt_mask, answer_mask]).unsqueeze(0)
        else:
            full_mask = torch.ones_like(full_ids)

        result = dict(data_dict_q)
        result["input_ids"] = full_ids
        result["labels"] = labels
        result["attention_mask"] = full_mask

        return result


# ============================================================================
# PART 4: INTEGRATION GUIDE
# ============================================================================

INTEGRATION_GUIDE = """
=== HOW TO INTEGRATE WITH SHOWUI TRAINING ===

1. Convert the data:
   python screen2ax_showui.py --convert --output_dir /path/to/data/Screen2AX

2. Copy this file into ShowUI's data/ directory:
   cp screen2ax_showui.py ShowUI/data/dset_screen2ax.py

3. Register the dataset in ShowUI/data/dataset.py:
   In the HybridDataset.__init__() method, add a branch:

       elif "screen2ax" in dataset:
           from data.dset_screen2ax import ElementDetectionDataset
           self.dataset = ElementDetectionDataset(
               dataset_dir=dataset_dir,
               dataset=dataset,
               json_data=json_data,
               processor=processor,
               inference=inference,
               args_dict=args_dict,
           )

4. Launch training:
   deepspeed --num_gpus=1 train.py \\
       --model_id showlab/ShowUI-2B \\
       --dataset_dir ./Screen2AX \\
       --train_dataset screen2ax \\
       --train_json hf_train \\
       --val_dataset screen2ax \\
       --val_json hf_val \\
       --epochs 10 \\
       --lr 3e-4 \\
       --lora_r 8 \\
       --batch_size 1 \\
       --max_new_tokens 2048 \\
       --max_visual_tokens 1280

   NOTE: --max_new_tokens should be high (2048+) since the model needs to
   output ALL elements at once. Some Screen2AX screenshots have 50-100 elements,
   producing long answer sequences.

5. Inference:
   from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

   model = Qwen2VLForConditionalGeneration.from_pretrained("your-finetuned-model")
   processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

   messages = [{
       "role": "user",
       "content": [
           {"type": "text", "text": "Detect all interactive UI elements with their types and bounding boxes. The bounding box coordinates [x1, y1, x2, y2] are relative to the screenshot, scaled from 0 to 1000. Output format: Type [x1, y1, x2, y2]; Type [x1, y1, x2, y2]; ..."},
           {"type": "image", "image": "screenshot.png"},
       ],
   }]
   # ... standard Qwen2-VL generation code ...

   # Parse output:
   import re
   elements = re.findall(r'(AX\w+)\s+\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', output_text)
   for role, x1, y1, x2, y2 in elements:
       print(f"{role}: [{int(x1)/1000:.3f}, {int(y1)/1000:.3f}, {int(x2)/1000:.3f}, {int(y2)/1000:.3f}]")

=== KEY DESIGN DECISIONS ===

- Coordinates use 0-1000 integer scale (not 0-1 floats) because:
  * Shorter token sequences (3-4 digits vs 4-6 chars per coordinate)
  * Consistent with ShowUI's xy_int=True mode
  * Easier for the model to learn discrete integer outputs

- Elements separated by semicolons for clear parsing

- Category names use raw macOS AX roles (AXButton, AXTextArea, etc.)
  which are directly useful for accessibility metadata generation

- The answer can get very long for complex UIs (50-100 elements).
  Consider:
  * Increasing --max_new_tokens to 2048-4096
  * Using --max_elements to cap during training
  * Filtering tiny/degenerate elements during conversion
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screen2AX → ShowUI conversion")
    parser.add_argument("--convert", action="store_true", help="Run conversion")
    parser.add_argument("--output_dir", default="./Screen2AX", help="Output directory")
    parser.add_argument("--min_elements", type=int, default=1)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--guide", action="store_true", help="Print integration guide")
    args = parser.parse_args()

    if args.guide:
        print(INTEGRATION_GUIDE)
    elif args.convert:
        convert_screen2ax(args.output_dir, args.min_elements, args.splits)
    else:
        parser.print_help()
        print("\n\nUse --convert to convert data, or --guide for integration instructions.")
