#!/bin/bash
# End-to-end: detect -> compose -> evaluate for the YOLO two-stage pipeline.
#
# Writes:
#   $OUT/detections/      per-image JSON with raw YOLO boxes (0-1000 space)
#   $OUT/val/parsed/      linearized AX trees (same contract as VLM preds)
#   $OUT/val/metadata/    per-sample counts/depth
#   $OUT/val/metrics*.json  evaluate.py output
#
# Usage:
#   bash yolo_compose/run_yolo_eval.sh            # defaults
#   SPLIT=train bash yolo_compose/run_yolo_eval.sh
#   STAGE1_CONF=0.35 OVERLAP=0.8 bash yolo_compose/run_yolo_eval.sh
set -euo pipefail

# ---- knobs (override via env) ----
SPLIT="${SPLIT:-val}"
STAGE1_WEIGHTS="${STAGE1_WEIGHTS:-/workspace/runs/yolo/stage1_elements/weights/best.pt}"
STAGE2_WEIGHTS="${STAGE2_WEIGHTS:-/workspace/runs/yolo/stage2_groups/weights/best.pt}"
IMAGES_DIR="${IMAGES_DIR:-/workspace/screen2ax_eval/data/screen2ax_linearized_simple/images}"
SPLIT_INFO="${SPLIT_INFO:-/workspace/data/yolo/split_info.json}"

STAGE1_CONF="${STAGE1_CONF:-0.25}"
STAGE2_CONF="${STAGE2_CONF:-0.25}"
NMS_IOU="${NMS_IOU:-0.45}"
MAX_DET="${MAX_DET:-500}"
IMGSZ="${IMGSZ:-1280}"
DEVICE="${DEVICE:-0}"

OVERLAP="${OVERLAP:-0.9}"
DEDUPE_IOU="${DEDUPE_IOU:-0.95}"

GT_KIND="${GT_KIND:-silver}"   # silver | human
if [ "$GT_KIND" = "silver" ]; then
  GT_DIR="/workspace/screen2ax_eval/results/qwen3vl_230b_simple/parsed"
else
  GT_DIR="/workspace/screen2ax_eval/data/screen2ax_linearized_simple/annotations"
fi

RUN_NAME="${RUN_NAME:-yolo_composed_s1-${STAGE1_CONF}_s2-${STAGE2_CONF}_ov-${OVERLAP}}"
OUT_ROOT="${OUT_ROOT:-/workspace/inference_results_yolo_composed}"
OUT="$OUT_ROOT/$RUN_NAME"
DET_DIR="$OUT/detections"
SPLIT_OUT="$OUT/$SPLIT"

echo "=== YOLO two-stage -> AX tree evaluation ==="
echo "Run:        $RUN_NAME"
echo "Split:      $SPLIT"
echo "GT:         $GT_KIND ($GT_DIR)"
echo "Stage1 conf: $STAGE1_CONF  Stage2 conf: $STAGE2_CONF  NMS IoU: $NMS_IOU  max_det: $MAX_DET  imgsz: $IMGSZ"
echo "Overlap:    $OVERLAP  Dedupe IoU: $DEDUPE_IOU"
echo "Output:     $OUT"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 1. detect ----
python "$SCRIPT_DIR/infer_yolo_two_stage.py" \
  --stage1-weights "$STAGE1_WEIGHTS" \
  --stage2-weights "$STAGE2_WEIGHTS" \
  --images-dir "$IMAGES_DIR" \
  --split-info "$SPLIT_INFO" \
  --split "$SPLIT" \
  --detections-dir "$DET_DIR" \
  --stage1-conf "$STAGE1_CONF" \
  --stage2-conf "$STAGE2_CONF" \
  --nms-iou "$NMS_IOU" \
  --max-det "$MAX_DET" \
  --imgsz "$IMGSZ" \
  --device "$DEVICE" \
  ${RESUME:+--resume}

# ---- 2. compose ----
python "$SCRIPT_DIR/compose_tree.py" \
  --detections-dir "$DET_DIR" \
  --output-dir "$SPLIT_OUT" \
  --overlap-threshold "$OVERLAP" \
  --dedupe-iou "$DEDUPE_IOU"

# ---- 3. evaluate ----
METRICS="$SPLIT_OUT/metrics_${GT_KIND}.json"
python /workspace/screen2ax_eval/evaluate.py \
  --gt-dir "$GT_DIR" \
  --pred-dir "$SPLIT_OUT/parsed" \
  --output "$METRICS" \
  --per-sample \
  --skip-ged \
  --simplified-roles \
  --workers 4

echo
echo "Metrics written to: $METRICS"
