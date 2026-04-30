#!/usr/bin/env bash
# Serve a Qwen3-VL model via vLLM (OpenAI-compatible) on 2 GPUs, port 8000.
# Required env vars: HF_TOKEN
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN}"
MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
GPUS="${GPUS:-device=2,3}"

docker run --runtime nvidia --gpus "\"$GPUS\"" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model "$MODEL" \
  --trust-remote-code --tensor-parallel-size 2 \
  --max-model-len 32768 --gpu-memory-utilization 0.9
