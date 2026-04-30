# Trained models

Model weights are not committed to this repository. Final checkpoints used in the thesis will be uploaded to HuggingFace Hub under the author's namespace.

## Models trained for the thesis

| Model                                                | Size  | Target HF Hub repo                          |
|------------------------------------------------------|-------|---------------------------------------------|
| Qwen3-VL-2B + Screen2AX silver LoRA                  | ~200M | `<hf-user>/qwen3-vl-2b-screen2ax-lora`      |
| Qwen3-VL-4B + Screen2AX silver LoRA                  | ~300M | `<hf-user>/qwen3-vl-4b-screen2ax-lora`      |
| Qwen3-VL-8B + Screen2AX silver LoRA                  | ~500M | `<hf-user>/qwen3-vl-8b-screen2ax-lora`      |
| YOLOv11l, leaf-element detector (silver-trained)     | ~50M  | `<hf-user>/yolov11l-screen2ax-elements`     |
| YOLOv11l, AXGroup detector (silver-trained)          | ~50M  | `<hf-user>/yolov11l-screen2ax-groups`       |

The Qwen3-VL-235B-A22B-Instruct teacher used to generate the silver corpus is the off-the-shelf `Qwen/Qwen3-VL-235B-A22B-Instruct`.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

base = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
model = PeftModel.from_pretrained(base, "<hf-user>/qwen3-vl-8b-screen2ax-lora")
```

For the YOLO weights, use `ultralytics` directly:

```python
from ultralytics import YOLO
elements = YOLO("<hf-user>/yolov11l-screen2ax-elements")
groups   = YOLO("<hf-user>/yolov11l-screen2ax-groups")
```
