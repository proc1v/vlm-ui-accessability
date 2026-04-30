#!/bin/bash
# LoRA fine-tuning of Qwen3-VL-2B-Instruct on Screen2AX silver data.
#
# Dataset: /workspace/data/qwen_sft/{train,val}.json  (LLaVA format)
#   - 1005 train / 112 val samples
#   - Silver targets: linearized macOS AX trees from Qwen3-VL-235B
#   - System prompt is baked into src/constants.py (SYSTEM_MESSAGE)
# Hardware: 2x NVIDIA B200 (Blackwell) on GPUs 3,4. Requires PyTorch cu128.

MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

# 2 GPUs * batch 4 * grad_accum 4 = effective batch 32
GLOBAL_BATCH_SIZE=32
BATCH_PER_DEVICE=4
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# Qwen3-VL uses 32x32 patches -> pixels = N * 32 * 32.
IMG_MIN_PIXELS=$((256 * 32 * 32))
IMG_MAX_PIXELS=$((1344 * 32 * 32))

OUTPUT_DIR=/workspace/trained_models/qwen3vl_2b_screen2ax_lora

deepspeed --include localhost:3,4 --master_port 29502 src/train/train_sft.py \
    --use_liger_kernel False \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /workspace/data/qwen_sft/train.json \
    --image_folder /workspace/data/qwen_sft \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $IMG_MIN_PIXELS \
    --image_max_pixels $IMG_MAX_PIXELS \
    --learning_rate 1e-5 \
    --merger_lr 0 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --max_grad_norm 0.5 \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 5 \
    --dataloader_num_workers 4
