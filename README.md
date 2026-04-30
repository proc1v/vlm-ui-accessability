# Vision-Language Models for Hierarchical Accessibility Metadata Generation from User Interface Screenshots

Code accompanying the master's thesis *Vision-Language Models for Hierarchical Accessibility Metadata Generation from User Interface Screenshots* by Nazar Protsiv (Ukrainian Catholic University, Faculty of Applied Sciences, 2026), supervised by Maksym Shamrai. The thesis investigates whether modern open-weight vision-language models can generate hierarchical macOS accessibility trees directly from screenshots at a quality competitive with the specialized Screen2AX pipeline. Builds on [Screen2AX (MacPaw)](https://github.com/MacPaw/Screen2AX), [OmniParser (Microsoft)](https://github.com/microsoft/OmniParser), and [ShowUI](https://github.com/showlab/ShowUI) / Qwen2-VL.

## What's in this repo

This repository contains **only original code authored for the thesis**. All upstream projects are pulled in as pinned git submodules under `third_party/`, with any local modifications captured as patches under `patches/`.

```
src/
  screen2ax_eval/   Zero-shot eval pipeline (vLLM OpenAI API client, AX-tree parser, metrics, Gradio viz)
  silver_to_yolo/   Convert Screen2AX silver-standard data to YOLO detection format + Qwen SFT JSON
  yolo_compose/     Two-stage pipeline: YOLO detector + Qwen describer composing AX trees
  training/         Qwen2-VL / ShowUI fine-tuning on Screen2AX-Tree (DeepSpeed + LoRA)
  baselines/        Zero-shot ShowUI runners
  prepare/          HuggingFace dataset preprocessing
  tools/            Dataset analysis, prediction visualization, Gradio app

scripts/            Entry-point shell scripts (train, infer, eval, vLLM serve)
configs/            DeepSpeed configs (zero1/zero2/zero3)
third_party/        Pinned upstream submodules — see EXTERNAL_REPOS.md
patches/            Local modifications to upstream repos (apply after cloning submodules)
docs/               Reproducibility, dataset, model docs + working notes from development
```

## Quick start

```bash
# 1. Clone with submodules
git clone --recurse-submodules <this-repo-url> thesis-screen2ax
cd thesis-screen2ax

# 2. Apply local patches to upstream code (only if you need our exact reproductions)
git -C third_party/OmniParser       apply ../../patches/omniparser-b0d5c9f5.patch
git -C third_party/Qwen-VL-Finetune apply ../../patches/qwen-vl-finetune-130ad7cc.patch
cp patches/omniparser-untracked/run_batch.py third_party/OmniParser/
cp patches/qwen-vl-finetune-untracked/scripts/* third_party/Qwen-VL-Finetune/scripts/

# 3. Install (separate envs per pipeline — see docs/reproducibility.md)
pip install -r requirements.txt

# 4. Pull data and trained model weights from HuggingFace Hub
#    (see docs/datasets.md and docs/models.md)
```

## Reproducing the thesis results

| Pipeline                       | Entry point                                       |
|--------------------------------|---------------------------------------------------|
| Zero-shot Qwen3-VL eval        | `scripts/eval_screen2ax.sh`                       |
| Qwen2-VL LoRA fine-tune        | `scripts/train_qwen_lora.sh`                      |
| Inference w/ fine-tuned model  | `scripts/infer_tree.sh <ckpt_dir>`                |
| Serve a model via vLLM         | `scripts/serve_vllm.sh`                           |
| YOLO + Qwen two-stage          | `src/yolo_compose/run_yolo_eval.sh`               |

Detailed environment setup, dataset acquisition, and full command sequences live in [`docs/reproducibility.md`](docs/reproducibility.md).

## License

Original code in this repository is released under the MIT License (see `LICENSE`). Submodules under `third_party/` retain their respective upstream licenses.
