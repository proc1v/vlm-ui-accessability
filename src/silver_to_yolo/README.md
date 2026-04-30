# silver_to_yolo

Convert linearized silver AX tree annotations (from the `screen2ax_eval`
pipeline) into two YOLO-format detection datasets:

1. **Stage 1 — Element Detection**: leaf nodes only, 6 classes
   (`AXButton`, `AXStaticText`, `AXImage`, `AXLink`, `AXTextArea`, `AXDisclosureTriangle`).
   `AXGroup` is excluded in both the leaf and container cases.
2. **Stage 2 — Group Detection**: `AXGroup` nodes with `>=1` child, 1 class.

Both stages share the same train/val split — saved to `split_info.json` —
so they stay aligned for the hybrid two-stage pipeline.

## Usage

```bash
# Both stages
python convert.py \
  --silver-dir /workspace/screen2ax_eval/results/qwen3vl_230b_simple/parsed \
  --images-dir /workspace/screen2ax_eval/data/screen2ax_linearized_simple/images \
  --output-dir /workspace/data/yolo \
  --val-ratio 0.1 --seed 42

# Single stage
python convert.py --stage 1 ...
python convert.py --stage 2 ...

# App-aware split (avoids leakage when filenames are like "appname_001.txt")
python convert.py --split-by-app --app-delimiter "_" ...

# Validate + optional visual spot-check
python validate.py --dataset-dir /workspace/data/yolo/stage1_detection
python validate.py --dataset-dir /workspace/data/yolo/stage2_grouping --visualize 5
```

## Output layout

```
output_dir/
├── stage1_detection/
│   ├── images/{train,val}/   # symlinks back to the original images
│   ├── labels/{train,val}/   # YOLO .txt files
│   └── dataset.yaml
├── stage2_grouping/
│   └── ... (same layout)
└── split_info.json
```

## Coordinate conversion

Silver annotations use `[x1, y1, x2, y2]` in the 0–1000 range. YOLO expects
`x_center y_center width height` in 0–1. Coords are clamped to `[0, 1000]`
before conversion and degenerate boxes (`x2 <= x1` or `y2 <= y1`) are skipped.
Boxes whose normalized width or height falls at/below `--min-bbox-size`
(default `0.001`) are also dropped.

## Notes

- Silver files are matched to images by stem (`0.txt` ↔ `0.png`). PNG and JPG
  are both supported.
- Images are symlinked to avoid duplication; the script falls back to copy
  across filesystems.
- Leaf detection uses parsed children (`len(node.children) == 0`), not raw
  indentation.
- An `AXGroup` with zero children is excluded from **both** stages — Stage 1
  excludes all groups, Stage 2 only keeps groups with `>=1` child.
