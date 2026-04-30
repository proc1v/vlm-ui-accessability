#!/usr/bin/env bash
# Fine-tune Qwen2-VL-7B on Screen2AX-Tree (8 GPUs, DeepSpeed Zero2 + LoRA).
# Required env vars: WANDB_KEY, _DATA_DIR, _SAVE_DIR
set -euo pipefail
: "${WANDB_KEY:?set WANDB_KEY}"
: "${_DATA_DIR:?set _DATA_DIR (dataset root)}"
: "${_SAVE_DIR:?set _SAVE_DIR (checkpoint output root)}"

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 1224 src/training/train.py \
  --wandb_key="$WANDB_KEY" \
  --model_id='Qwen/Qwen2-VL-7B-Instruct' \
  --version='Qwen/Qwen2-VL-7B-Instruct' \
  --dataset_dir="$_DATA_DIR" \
  --log_base_dir="$_SAVE_DIR" \
  --epochs=5 --batch_size=1 --grad_accumulation_steps=4 \
  --model_max_length=8192 \
  --exp_id="screen2ax_tree_qwen2_7b" \
  --train_dataset="screen2ax_tree" --train_json="train" \
  --val_dataset="screen2ax_tree"   --val_json="val" \
  --precision="bf16" --attn_imple="sdpa" --workers=6 \
  --lora_r=32 --lora_alpha=64 \
  --min_visual_tokens=256 --max_visual_tokens=1344 --max_new_tokens=2048 \
  --lr=0.0001 --ds_zero="zero2" --gradient_checkpointing \
  --lm_skip_ratio=0.0 --random_sample --steps_per_epoch=1000
