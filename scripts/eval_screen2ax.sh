#!/usr/bin/env bash
# Full Screen2AX zero-shot eval pipeline (linearize -> vLLM inference -> metrics -> visualize).
# Assumes a vLLM OpenAI-compatible server is already running (see scripts/serve_vllm.sh).
set -euo pipefail
DATA_DIR="${DATA_DIR:-./data/screen2ax_linearized}"
RUN_DIR="${RUN_DIR:-./results/run}"

# 1. Linearize HF Screen2AX-Tree dataset (skip if already done)
[ -d "$DATA_DIR" ] || \
  python src/screen2ax_eval/linearize_screen2ax.py export --output-dir "$DATA_DIR"

# 2. Inference via vLLM
python src/screen2ax_eval/inference.py \
  --dataset-dir "$DATA_DIR" \
  --output-dir "$RUN_DIR"

# 3. Metrics
python src/screen2ax_eval/evaluate.py \
  --gt-dir "$DATA_DIR/annotations" \
  --pred-dir "$RUN_DIR/parsed" \
  --output  "$RUN_DIR/metrics.json" \
  --per-sample --skip-ged
