#!/usr/bin/env python3
"""Compute Screen2AX per-class detection metrics on saved predictions.

Metrics (per class, IoU threshold 0.5):
- Precision, Recall, F1
- AP50 (VOC-style area under PR curve at IoU >= 0.5)

Detection classes (from the Screen2AX paper): AXButton, AXDisclosureTriangle,
AXImage, AXLink, AXTextArea.

This is a *detection-only* evaluator — tree structure is ignored. Each node is
treated as a standalone (role, bbox) detection. Confidence scores are not
available from text predictions, so AP is computed assuming equal confidence
per prediction (predictions sorted arbitrarily, which reduces AP to the P/R
operating point's area; we emit the AP derived from the greedy matching curve).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from utils.parser import AXNode, parse_tree
from utils.metrics import compute_iou

logger = logging.getLogger(__name__)

# Detection classes from the Screen2AX paper
DETECTION_CLASSES: List[str] = [
    "AXButton",
    "AXDisclosureTriangle",
    "AXImage",
    "AXLink",
    "AXTextArea",
]


# ---------------------------------------------------------------------------
# Per-sample matching: greedy assignment for a single class
# ---------------------------------------------------------------------------


def _match_detections(
    gt_boxes: List[List[int]],
    pred_boxes: List[List[int]],
    iou_threshold: float = 0.5,
) -> Tuple[List[bool], List[float], int]:
    """Greedy-match predictions to GT for one (sample, class) pair.

    Returns:
        matched: list of len(pred_boxes), True if the pred is a TP (matched
                 to an unused GT with IoU >= threshold).
        ious: IoU of each pred with its best GT (0 if no GT).
        n_gt: number of GT boxes.
    """
    n_gt = len(gt_boxes)
    matched = [False] * len(pred_boxes)
    ious = [0.0] * len(pred_boxes)

    if n_gt == 0 or not pred_boxes:
        return matched, ious, n_gt

    # Build IoU matrix, then greedily pick best pairs.
    iou_matrix = np.zeros((len(pred_boxes), n_gt))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pb, gb)

    used_gt = set()
    # Process predictions in order — callers should pre-sort by confidence
    # (here: original parse order, since text preds have no confidences).
    for i in range(len(pred_boxes)):
        best_j = -1
        best_iou = iou_threshold
        for j in range(n_gt):
            if j in used_gt:
                continue
            if iou_matrix[i, j] >= best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        ious[i] = float(iou_matrix[i].max()) if n_gt else 0.0
        if best_j >= 0:
            matched[i] = True
            used_gt.add(best_j)

    return matched, ious, n_gt


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------


def _eval_sample(
    gt_text: str,
    pred_text: str,
    iou_threshold: float = 0.5,
    role_mapping: Optional[Dict[str, Optional[str]]] = None,
    classes: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Evaluate one sample, returning per-class match info.

    For each class:
        {"matches": List[bool], "ious": List[float], "n_gt": int}
    """
    classes = classes or DETECTION_CLASSES
    gt_nodes = parse_tree(gt_text, role_mapping=role_mapping)
    pred_nodes = parse_tree(pred_text, role_mapping=role_mapping)

    per_class: Dict[str, Dict] = {}
    for cls in classes:
        gt_boxes = [n.bbox for n in gt_nodes if n.role == cls and n.bbox is not None]
        pred_boxes = [n.bbox for n in pred_nodes if n.role == cls and n.bbox is not None]
        matches, ious, n_gt = _match_detections(gt_boxes, pred_boxes, iou_threshold)
        per_class[cls] = {"matches": matches, "ious": ious, "n_gt": n_gt}
    return per_class


def _summarize_sample(per_class: Dict[str, Dict]) -> Dict:
    """Collapse per-class match lists into per-sample TP/FP/FN + P/R/F1."""
    tp = fp = fn = n_gt = n_pred = 0
    for info in per_class.values():
        matches = info["matches"]
        s_tp = sum(1 for m in matches if m)
        s_pred = len(matches)
        tp += s_tp
        fp += s_pred - s_tp
        fn += info["n_gt"] - s_tp
        n_gt += info["n_gt"]
        n_pred += s_pred

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _eval_one_job(job: Dict) -> Dict:
    """Worker entry point for a single sample (picklable)."""
    stem = job["stem"]
    gt_path = Path(job["gt_path"])
    pred_path = Path(job["pred_path"])
    try:
        gt_text = gt_path.read_text(encoding="utf-8")
        pred_text = pred_path.read_text(encoding="utf-8")
        role_mapping = (
            config.SIMPLIFIED_ROLE_MAPPING if job["simplified_roles"] else None
        )
        per_class = _eval_sample(
            gt_text,
            pred_text,
            iou_threshold=job["iou_threshold"],
            role_mapping=role_mapping,
            classes=job["classes"],
        )
        sample_summary = _summarize_sample(per_class)
        # Also compute compact per-class summary (TP/FP/FN only — drop raw lists)
        per_class_compact = {
            cls: {
                "tp": sum(1 for m in info["matches"] if m),
                "fp": len(info["matches"]) - sum(1 for m in info["matches"] if m),
                "fn": info["n_gt"] - sum(1 for m in info["matches"] if m),
                "n_gt": info["n_gt"],
                "n_pred": len(info["matches"]),
            }
            for cls, info in per_class.items()
        }
        return {
            "filename": stem,
            "per_class": per_class,
            "sample_summary": sample_summary,
            "per_class_summary": per_class_compact,
        }
    except Exception as e:
        logger.error(f"Error evaluating {stem}: {e}", exc_info=True)
        return {"filename": stem, "per_class": {}, "error": str(e)}


# ---------------------------------------------------------------------------
# Aggregation: pool across samples → P, R, F1, AP50 per class
# ---------------------------------------------------------------------------


def _compute_ap(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    """VOC-style AP: area under the precision-recall curve.

    Uses the 11-point interpolation (PASCAL VOC 2007) for stability with small
    sample counts. Assumes tp/fp arrays are ordered by descending confidence.
    """
    if n_gt == 0:
        return 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / n_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1)

    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        mask = recalls >= t
        p = precisions[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return float(ap)


def _aggregate(
    results: List[Dict],
    classes: List[str],
) -> Dict:
    """Pool per-sample matches into per-class P/R/F1/AP."""
    per_class_out: Dict[str, Dict] = {}

    for cls in classes:
        # Pool predictions (matches + IoU as proxy confidence) and GT counts.
        all_matches: List[bool] = []
        all_ious: List[float] = []
        total_gt = 0
        for r in results:
            info = r.get("per_class", {}).get(cls)
            if not info:
                continue
            all_matches.extend(info["matches"])
            all_ious.extend(info["ious"])
            total_gt += info["n_gt"]

        n_pred = len(all_matches)
        tp_total = sum(1 for m in all_matches if m)
        fp_total = n_pred - tp_total
        fn_total = total_gt - tp_total

        precision = tp_total / n_pred if n_pred > 0 else 0.0
        recall = tp_total / total_gt if total_gt > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # AP: sort by descending IoU as a proxy for confidence (text preds
        # lack real scores — this gives a sensible PR curve where higher-IoU
        # matches count first).
        if n_pred > 0 and total_gt > 0:
            order = np.argsort(-np.array(all_ious))
            sorted_matches = np.array(all_matches)[order]
            tp_arr = sorted_matches.astype(np.int32)
            fp_arr = 1 - tp_arr
            ap50 = _compute_ap(tp_arr, fp_arr, total_gt)
        else:
            ap50 = 0.0

        per_class_out[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "ap50": round(ap50, 4),
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "n_gt": total_gt,
            "n_pred": n_pred,
        }

    # Macro-average across classes (unweighted)
    def _mean(key: str) -> float:
        vals = [per_class_out[c][key] for c in classes if per_class_out[c]["n_gt"] > 0]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    return {
        "per_class": per_class_out,
        "macro": {
            "precision": _mean("precision"),
            "recall": _mean("recall"),
            "f1": _mean("f1"),
            "ap50": _mean("ap50"),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_evaluation(args: argparse.Namespace) -> None:
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    if not gt_dir.exists():
        logger.error(f"GT directory not found: {gt_dir}")
        return
    if not pred_dir.exists():
        logger.error(f"Predictions directory not found: {pred_dir}")
        return

    gt_files = sorted(gt_dir.glob("*.txt"))
    if not gt_files:
        logger.error(f"No .txt files found in {gt_dir}")
        return
    if args.num_samples is not None:
        gt_files = gt_files[: args.num_samples]

    classes = args.classes if args.classes else DETECTION_CLASSES

    jobs: List[Dict] = []
    n_missing = 0
    for gt_path in gt_files:
        stem = gt_path.stem
        pred_path = pred_dir / f"{stem}.txt"
        if not pred_path.exists():
            n_missing += 1
            continue
        jobs.append({
            "stem": stem,
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
            "iou_threshold": args.iou_threshold,
            "simplified_roles": args.simplified_roles,
            "classes": classes,
        })

    logger.info(f"Evaluating {len(jobs)} samples ({n_missing} missing predictions)")

    results: List[Dict] = []
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_eval_one_job, j) for j in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Detecting"):
                results.append(fut.result())
    else:
        for job in tqdm(jobs, desc="Detecting"):
            results.append(_eval_one_job(job))

    summary = _aggregate(results, classes)
    summary["n_total"] = len(jobs)
    summary["n_missing"] = n_missing
    summary["iou_threshold"] = args.iou_threshold
    summary["model"] = args.model or config.MODEL_NAME

    # Pretty-print
    print(f"\n{'='*70}")
    print(f" Screen2AX Detection Metrics (IoU >= {args.iou_threshold})")
    print(f"{'='*70}")
    print(f"Model: {summary['model']}")
    print(f"Samples: {summary['n_total']}" + (f"  (missing preds: {n_missing})" if n_missing else ""))
    print()
    header = (
        f"{'Class':<24} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AP50':>7} "
        f"{'TP':>7} {'FP':>7} {'FN':>7} {'#GT':>7} {'#Pred':>7}"
    )
    print(header)
    print("\u2500" * len(header))

    total_tp = total_fp = total_fn = total_gt = total_pred = 0
    for cls in classes:
        r = summary["per_class"][cls]
        print(
            f"{cls:<24} {r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f} "
            f"{r['ap50']:>7.3f} {r['tp']:>7} {r['fp']:>7} {r['fn']:>7} "
            f"{r['n_gt']:>7} {r['n_pred']:>7}"
        )
        total_tp += r["tp"]
        total_fp += r["fp"]
        total_fn += r["fn"]
        total_gt += r["n_gt"]
        total_pred += r["n_pred"]

    # Micro over all classes (pooled counts)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0
        else 0.0
    )

    m = summary["macro"]
    print("\u2500" * len(header))
    print(
        f"{'macro avg':<24} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
        f"{m['ap50']:>7.3f} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7}"
    )
    print(
        f"{'micro avg':<24} {micro_p:>7.3f} {micro_r:>7.3f} {micro_f1:>7.3f} "
        f"{'-':>7} {total_tp:>7} {total_fp:>7} {total_fn:>7} "
        f"{total_gt:>7} {total_pred:>7}"
    )

    summary["micro"] = {
        "precision": round(micro_p, 4),
        "recall": round(micro_r, 4),
        "f1": round(micro_f1, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "n_gt": total_gt,
        "n_pred": total_pred,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Detection metrics saved to {output_path}")

    # Per-sample dump (for spotting best / worst examples)
    if args.per_sample and args.output:
        output_path = Path(args.output)
        per_sample_path = output_path.parent / (output_path.stem + "_per_sample.jsonl")
        # Sort by filename (numeric if possible) for deterministic ordering
        def _sort_key(r: Dict):
            try:
                return (0, int(r.get("filename", "")))
            except (TypeError, ValueError):
                return (1, r.get("filename", ""))
        sorted_results = sorted(results, key=_sort_key)
        with open(per_sample_path, "w", encoding="utf-8") as f:
            for r in sorted_results:
                row = {
                    "filename": r.get("filename"),
                    **r.get("sample_summary", {}),
                    "per_class": r.get("per_class_summary", {}),
                }
                if "error" in r:
                    row["error"] = r["error"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"Per-sample detection metrics saved to {per_sample_path}")

        # Print quick best/worst hit list to terminal
        scored = [
            r for r in sorted_results
            if r.get("sample_summary", {}).get("n_gt", 0) > 0
        ]
        scored.sort(key=lambda r: (r["sample_summary"]["f1"], -r["sample_summary"]["n_gt"]))
        top_n = min(args.top_n, len(scored))
        if top_n > 0:
            print(f"\nWorst {top_n} samples by F1:")
            print(f"  {'filename':<20} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'#GT':>5}")
            for r in scored[:top_n]:
                s = r["sample_summary"]
                print(
                    f"  {str(r['filename']):<20} {s['f1']:>6.3f} {s['tp']:>5} "
                    f"{s['fp']:>5} {s['fn']:>5} {s['n_gt']:>5}"
                )
            print(f"\nBest {top_n} samples by F1:")
            print(f"  {'filename':<20} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'#GT':>5}")
            for r in scored[-top_n:][::-1]:
                s = r["sample_summary"]
                print(
                    f"  {str(r['filename']):<20} {s['f1']:>6.3f} {s['tp']:>5} "
                    f"{s['fp']:>5} {s['fn']:>5} {s['n_gt']:>5}"
                )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute Screen2AX detection metrics (per-class P/R/F1/AP50)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gt-dir", type=str, required=True, help="Directory with GT .txt files")
    p.add_argument("--pred-dir", type=str, required=True, help="Directory with predicted .txt files")
    p.add_argument("--output", type=str, default=None, help="Path to save detection metrics JSON")
    p.add_argument("--num-samples", type=int, default=None, help="Evaluate only first N samples")
    p.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for a TP match")
    p.add_argument(
        "--simplified-roles",
        action="store_true",
        help="Apply 7-class simplified role mapping before evaluation",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help=f"Classes to evaluate (default: {' '.join(DETECTION_CLASSES)})",
    )
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    p.add_argument("--model", type=str, default=None, help="Model name for reporting")
    p.add_argument(
        "--per-sample",
        action="store_true",
        help="Save per-sample TP/FP/FN + P/R/F1 to {output_stem}_per_sample.jsonl",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Print N best/worst samples when --per-sample is set",
    )
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    run_evaluation(args)
