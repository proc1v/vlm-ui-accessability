# External repositories

This thesis builds on three upstream projects, included here as **pinned git submodules** under `third_party/`. Each submodule is checked out at the exact commit used during the thesis experiments.

| Submodule path                       | Upstream                                                      | Pinned commit | License        |
|--------------------------------------|---------------------------------------------------------------|---------------|----------------|
| `third_party/Screen2AX`              | https://github.com/MacPaw/Screen2AX                           | `c143022a`    | MIT            |
| `third_party/OmniParser`             | https://github.com/microsoft/OmniParser                       | `b0d5c9f5`    | CC-BY-4.0 / MIT |
| `third_party/Qwen-VL-Finetune`       | https://github.com/2U1/Qwen-VL-Series-Finetune                | `130ad7cc`    | Apache-2.0     |

## What each project provides

- **Screen2AX** — Original dataset and baseline for macOS AX-tree generation from screenshots. We use the Screen2AX-Tree HuggingFace dataset and re-implement evaluation in `src/screen2ax_eval/` for compatibility with vLLM-served models.
- **OmniParser** — Microsoft's icon/text detector for GUI screenshots. We use it as a strong detection baseline; our patches add a batch-inference runner and adjust `util/utils.py` for our evaluation harness.
- **Qwen-VL-Finetune** — Third-party LoRA fine-tuning code for Qwen2-VL / Qwen2.5-VL / Qwen3-VL. We use it for an alternative fine-tuning path (in addition to the ShowUI/DeepSpeed pipeline in `src/training/`); our patches add three Screen2AX-specific launch scripts.

## Local modifications

We made small local changes to two upstream repos. Diffs are captured under `patches/`:

```
patches/
├── omniparser-b0d5c9f5.patch                # ~1.5K lines — edits to demo.ipynb and util/utils.py
├── omniparser-untracked/run_batch.py        # New: batch inference driver
├── qwen-vl-finetune-130ad7cc.patch          # ~70 lines — requirements.txt + src/constants.py
└── qwen-vl-finetune-untracked/scripts/      # New: 3 Screen2AX fine-tune launchers
```

Apply after cloning submodules:

```bash
git -C third_party/OmniParser       apply ../../patches/omniparser-b0d5c9f5.patch
git -C third_party/Qwen-VL-Finetune apply ../../patches/qwen-vl-finetune-130ad7cc.patch
cp -r patches/omniparser-untracked/.       third_party/OmniParser/
cp -r patches/qwen-vl-finetune-untracked/. third_party/Qwen-VL-Finetune/
```

Screen2AX has no local changes — use it pristine.

## Citing the upstream work

Please cite the original papers / repos when using these components:

- Screen2AX — MacPaw research team, see `third_party/Screen2AX/CITATION.cff`.
- OmniParser — Lu et al., Microsoft Research; project page: https://microsoft.github.io/OmniParser/
- Qwen2-VL / Qwen2.5-VL / Qwen3-VL — Alibaba DAMO; the fine-tuning code is a community fork by [@2U1](https://github.com/2U1).
- ShowUI — Lin et al., CVPR 2025;
