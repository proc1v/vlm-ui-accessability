#!/usr/bin/env bash
# Run inference with a fine-tuned Qwen2-VL LoRA checkpoint on Screen2AX-Tree.
# Usage: ./scripts/infer_tree.sh <checkpoint_dir> [split=val] [output_dir=inference_results_tree]
set -euo pipefail
CHECKPOINT="${1:?path to ckpt_model dir}"
SPLIT="${2:-val}"
OUT="${3:-inference_results_tree}"

python src/training/infer_tree.py \
  --checkpoint "$CHECKPOINT" \
  --dataset_dir datasets/Screen2AX_linearized \
  --split "$SPLIT" \
  --output_dir "$OUT"
