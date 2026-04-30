# Screen2AX Zero-Shot VLM Evaluation Pipeline

Evaluate Vision-Language Models on the Screen2AX accessibility tree generation task.

## Setup

```bash
cd /workspace/screen2ax_eval
pip install -r requirements.txt
```

## Project Structure

```
screen2ax_eval/
├── config.py                # Shared configuration (paths, prompts, roles, colors)
├── linearize_screen2ax.py   # Convert HuggingFace dataset to normalized format
├── inference.py             # Run VLM inference via vLLM OpenAI-compatible API
├── evaluate.py              # Compute Screen2AX metrics on saved predictions
├── visualize.py             # Gradio app for side-by-side GT vs Pred comparison
└── utils/
    ├── parser.py            # Parse linearized AX tree format
    ├── metrics.py           # Screen2AX evaluation metrics (Edge F1, Leaves F1, CM, GED)
    └── normalize.py         # Coordinate normalization utilities
```

## Step 1: Prepare the Dataset

Download and linearize the Screen2AX-Tree dataset from HuggingFace.
This converts the raw accessibility trees into normalized format: `AXRole(subrole) [x1,y1,x2,y2]` with coordinates in 0-1000 range.

```bash
# Export full dataset (images + annotations + LLaVA JSON)
python linearize_screen2ax.py export --output-dir ./data/screen2ax_linearized

# Preview a few samples first (no files written)
python linearize_screen2ax.py preview --n 5

# Limit tree depth
python linearize_screen2ax.py export --output-dir ./data/screen2ax_linearized --max-depth 6
```

Output:
```
data/screen2ax_linearized/
├── images/          # Screenshot PNGs
├── annotations/     # Per-sample .txt GT files (normalized xyxy, AX-prefixed roles)
├── train.json       # LLaVA conversation format
├── val.json
└── config.json
```

## Step 2: Run Inference

Send screenshots to a VLM via the vLLM OpenAI-compatible API and save predictions.

**Prerequisites:** A running vLLM server (e.g., `vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct`).

```bash
# Run on full test set
python inference.py \
  --dataset-dir ./data/screen2ax_linearized \
  --output-dir ./results/qwen3vl_30b

# Run on a small subset for testing
python inference.py \
  --dataset-dir ./data/screen2ax_linearized \
  --output-dir ./results/qwen3vl_30b \
  --num-samples 10

# Custom vLLM endpoint
python inference.py \
  --base-url http://gpu-server:8000/v1 \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --output-dir ./results/qwen3vl_30b

# Resume an interrupted run (skips already-processed images)
python inference.py \
  --dataset-dir ./data/screen2ax_linearized \
  --output-dir ./results/qwen3vl_30b \
  --resume

# Concurrent requests for faster throughput
python inference.py \
  --dataset-dir ./data/screen2ax_linearized \
  --output-dir ./results/qwen3vl_30b \
  --batch-size 4
```

Output:
```
results/qwen3vl_30b/
├── raw/             # Raw model output (before cleaning)
├── parsed/          # Cleaned predictions (markdown fences stripped, only AX tree lines)
└── metadata/        # Per-sample JSON (latency, token counts, image dimensions, prompt)
```

### All inference flags

| Flag | Default | Description |
|---|---|---|
| `--dataset-dir` | `./data/screen2ax_linearized` | Path to dataset with `images/` and `annotations/` |
| `--output-dir` | `./results/default_run` | Where to save predictions |
| `--base-url` | `http://localhost:8000/v1` | vLLM server URL |
| `--model` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | Model name |
| `--api-key` | `OPENAI_API_KEY` env or `dummy` | API key |
| `--num-samples` | all | Limit number of samples |
| `--max-tokens` | 4096 | Max output tokens |
| `--temperature` | 0.0 | Sampling temperature |
| `--resume` | off | Skip already-processed images |
| `--batch-size` | 1 | Concurrent requests |
| `--verbose` | off | DEBUG logging |

## Step 3: Evaluate

Compute Screen2AX metrics comparing predictions against ground truth.

```bash
# Basic evaluation (skip GED for speed)
python evaluate.py \
  --gt-dir ./data/screen2ax_linearized/annotations \
  --pred-dir ./results/qwen3vl_30b/parsed \
  --output ./results/qwen3vl_30b/metrics.json \
  --skip-ged

# Full evaluation with GED
python evaluate.py \
  --gt-dir ./data/screen2ax_linearized/annotations \
  --pred-dir ./results/qwen3vl_30b/parsed \
  --output ./results/qwen3vl_30b/metrics.json

# Save per-sample results (needed for visualize.py sorting)
python evaluate.py \
  --gt-dir ./data/screen2ax_linearized/annotations \
  --pred-dir ./results/qwen3vl_30b/parsed \
  --output ./results/qwen3vl_30b/metrics.json \
  --per-sample \
  --skip-ged

# Quick check on first 10 samples
python evaluate.py \
  --gt-dir ./data/screen2ax_linearized/annotations \
  --pred-dir ./results/qwen3vl_30b/parsed \
  --num-samples 10 \
  --skip-ged
```

### GT format flags

If your GT annotations are **not** in normalized `[x1,y1,x2,y2]` format (e.g., produced by the original `linearize_screen2ax.py`):

```bash
# GT is in pixel [x,y,w,h] format (needs image dimensions to normalize)
python evaluate.py \
  --gt-dir ./annotations \
  --pred-dir ./results/run/parsed \
  --gt-format xywh \
  --no-gt-normalized \
  --metadata-dir ./results/run/metadata
```

Annotations produced by this project's `linearize_screen2ax.py` are already normalized xyxy, so no extra flags are needed.

### All evaluate flags

| Flag | Default | Description |
|---|---|---|
| `--gt-dir` | required | Directory with GT `.txt` files |
| `--pred-dir` | required | Directory with predicted `.txt` files (filenames must match) |
| `--output` | none | Path to save aggregated metrics JSON |
| `--per-sample` | off | Also save `_per_sample.jsonl` |
| `--num-samples` | all | Evaluate only first N |
| `--skip-ged` | off | Skip GED (much faster) |
| `--ged-timeout` | 30s | Timeout per sample for GED |
| `--iou-threshold` | 0.5 | IoU threshold for node matching |
| `--gt-format` | `xyxy` | GT bbox format: `xyxy` or `xywh` |
| `--no-gt-normalized` | off | GT coords are pixels, not 0-1000 |
| `--metadata-dir` | none | Metadata JSONs for image dimensions |
| `--model` | from config | Model name for the report |
| `--verbose` | off | DEBUG logging |

### Metrics

| Metric | Description |
|---|---|
| **Edge F1** | F1 on parent-child edges (after Hungarian node matching) |
| **Leaves F1** | F1 on edges where the child is a leaf node |
| **Complete Match** | 1.0 if all edges match exactly, 0.0 otherwise |
| **GED** | Graph Edit Distance (NP-hard, timeout-bounded) |

## Step 4: Visualize

Interactive Gradio app for side-by-side comparison of GT vs predicted bounding boxes.

```bash
# Basic usage
python visualize.py \
  --images-dir ./data/screen2ax_linearized/images \
  --gt-dir ./data/screen2ax_linearized/annotations \
  --pred-dir ./results/qwen3vl_30b/parsed

# With per-sample metrics (enables sorting by Edge F1, GED, etc.)
python visualize.py \
  --images-dir ./data/screen2ax_linearized/images \
  --gt-dir ./data/screen2ax_linearized/annotations \
  --pred-dir ./results/qwen3vl_30b/parsed \
  --metrics-file ./results/qwen3vl_30b/metrics_per_sample.jsonl

# Custom port
python visualize.py \
  --images-dir ./data/screen2ax_linearized/images \
  --gt-dir ./data/screen2ax_linearized/annotations \
  --pred-dir ./results/qwen3vl_30b/parsed \
  --port 7861

# Public share link
python visualize.py \
  --images-dir ./data/screen2ax_linearized/images \
  --gt-dir ./data/screen2ax_linearized/annotations \
  --pred-dir ./results/qwen3vl_30b/parsed \
  --share
```

Then open `http://localhost:7860` in your browser.

### Features

- **Side-by-side view**: GT bounding boxes on the left, predictions on the right
- **Role filtering**: Toggle visibility per role category (Containers, Text, Controls, Menus, Display, Layout, Other)
- **Sorting**: Sort samples by filename, Edge F1, GED, or node count
- **Per-sample metrics**: Shown next to the sample selector (requires `--metrics-file`)
- **Tree text**: Raw linearized tree text for both GT and prediction

## Full Pipeline Example

```bash
# 1. Prepare dataset
python linearize_screen2ax.py export --output-dir ./data/screen2ax_linearized

# 2. Run inference (assumes vLLM server is running)
python inference.py \
  --dataset-dir ./data/screen2ax_linearized \
  --output-dir ./results/qwen3vl_30b

# 3. Evaluate
python evaluate.py \
  --gt-dir ./data/screen2ax_linearized/annotations \
  --pred-dir ./results/qwen3vl_30b/parsed \
  --output ./results/qwen3vl_30b/metrics.json \
  --per-sample \
  --skip-ged

# 4. Visualize
python visualize.py \
  --images-dir ./data/screen2ax_linearized/images \
  --gt-dir ./data/screen2ax_linearized/annotations \
  --pred-dir ./results/qwen3vl_30b/parsed \
  --metrics-file ./results/qwen3vl_30b/metrics_per_sample.jsonl
```
