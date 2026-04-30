# Thesis-to-code mapping

Cross-reference between thesis chapters and the code that produced each result.

| Thesis section | Topic                                                 | Code path                                                                                  |
|----------------|-------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Ch. 4 §4.2.1   | Screen2AX-Tree dataset & linearization                 | `src/screen2ax_eval/linearize_screen2ax.py`                                                |
| Ch. 4 §4.2.2   | Ground-truth visual quality analysis                   | `src/tools/` (figure rendering)                                                            |
| Ch. 4 §4.2.3   | Silver-corpus construction (Qwen3-VL-235B teacher)     | `scripts/serve_vllm.sh`, `src/screen2ax_eval/inference.py`                                 |
| Ch. 4 §4.3.1   | Zero-shot Qwen3-VL prompting                           | `src/screen2ax_eval/inference.py`, `scripts/eval_screen2ax.sh`                             |
| Ch. 4 §4.3.2   | Supervised fine-tuning via LoRA (2B / 4B / 8B)         | `third_party/Qwen-VL-Finetune` + `patches/qwen-vl-finetune-untracked/scripts/`             |
| Ch. 4 §4.3.3   | Hybrid YOLOv11l + composition pipeline                 | `src/silver_to_yolo/`, `src/yolo_compose/`                                                 |
| Ch. 4 §4.4.1   | Intrinsic metrics (Edge F1, Leaves F1, GED, …)         | `src/screen2ax_eval/utils/metrics.py`, `src/screen2ax_eval/evaluate.py`                    |
| Ch. 4 §4.4.2   | Downstream Screen2AX-Task evaluation (GPT-5 selector)  | `src/screen2ax_eval/downstream/`                                                           |
| Ch. 5 §5.2     | Intrinsic results (Tables 5.1–5.2)                     | `src/screen2ax_eval/aggregate_all.py`                                                      |
| Ch. 5 §5.3     | Per-class detection (Table 5.3)                        | `src/screen2ax_eval/aggregate_all.py`                                                      |
| Ch. 5 §5.4     | Downstream results (Table 5.4)                         | `src/screen2ax_eval/downstream/`                                                           |
| Ch. 5 §5.5     | Qualitative figures                                    | `src/screen2ax_eval/render_hierarchy_figure.py`, `src/tools/`                              |

OmniParser is included as an external baseline (`third_party/OmniParser` + `patches/omniparser-untracked/run_batch.py`) and reported in Table 5.4 alongside the original Screen2AX pipeline (`third_party/Screen2AX`, used unmodified).
