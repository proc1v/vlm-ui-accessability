# Reproducibility

Environment, and command sequences used in the thesis experiments.

## Environments

The four pipelines have **incompatible Python dependencies** — use a separate virtualenv per pipeline:

| Pipeline                           | Suggested env name | Requirements file                                          |
|------------------------------------|--------------------|------------------------------------------------------------|
| Qwen3-VL LoRA fine-tuning          | `.venv-qwen`       | `requirements.txt`                                         |
| Screen2AX zero-shot eval (vLLM)    | `.venv-eval`       | `src/screen2ax_eval/requirements.txt`                      |
| YOLOv11l detection                 | `.venv-yolo`       | (ultralytics)                                              |
| OmniParser                         | `.venv-omniparser` | `third_party/OmniParser/requirements.txt`                  |

```bash
python -m venv .venv-qwen && source .venv-qwen/bin/activate && pip install -r requirements.txt
```

## Datasets

See [`datasets.md`](datasets.md) for download/preparation steps for:

- Screen2AX-Tree (HuggingFace) — primary
- Screen2AX-Task (HuggingFace) — downstream grounding benchmark
- Silver corpus (derived via `src/screen2ax_eval/inference.py` against the 235B teacher)
- YOLO-formatted Screen2AX (derived via `src/silver_to_yolo/`)
- Qwen SFT JSON (derived via `src/silver_to_yolo/build_qwen_sft_json.py`)

## Trained models

See [`models.md`](models.md) for HuggingFace Hub links to the LoRA adapters and YOLO weights produced by this work.

## End-to-end reproduction

```bash
# Set env vars
export WANDB_KEY=<your-wandb-key>
export HF_TOKEN=<your-hf-token>
export _DATA_DIR=/path/to/datasets
export _SAVE_DIR=/path/to/checkpoints

# 1. Download + linearize Screen2AX-Tree (one-time)
python src/screen2ax_eval/linearize_screen2ax.py export \
  --output-dir ./data/screen2ax_linearized

# 2. Generate silver corpus with the Qwen3-VL-235B-A22B-Instruct teacher
./scripts/serve_vllm.sh &              # in a separate terminal
python src/screen2ax_eval/inference.py \
  --input ./data/screen2ax_linearized \
  --output ./data/silver

# 3. Zero-shot eval of the Qwen3-VL family (2B / 4B / 8B) via vLLM
./scripts/eval_screen2ax.sh

# 4. Fine-tune Qwen3-VL on silver corpus (LoRA, ZeRO-2)
./scripts/train_qwen_lora.sh

# 5. Inference with fine-tuned model
./scripts/infer_tree.sh "$_SAVE_DIR/screen2ax_qwen3_8b/ckpt_model"
```