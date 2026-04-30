#!/usr/bin/env python3
"""Compute Screen2AX metrics on saved predictions."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from utils.parser import parse_tree
from utils.normalize import xywh_to_xyxy, normalize_bbox
from utils.metrics import evaluate_sample, evaluate_batch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GT loading with format conversion
# ---------------------------------------------------------------------------


def load_annotation(
    path: Path,
    gt_format: str = "xyxy",
    gt_normalized: bool = True,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
    normalize_range: int = 1000,
) -> str:
    """Load a GT annotation file and optionally convert coordinates.

    Args:
        path: Path to the .txt annotation file.
        gt_format: "xyxy" or "xywh" — the bbox format in the file.
        gt_normalized: If True, coords are already in normalized 0-1000 range.
            If False, coords are in pixel space and will be normalized.
        img_width: Image width in pixels (needed if gt_normalized=False).
        img_height: Image height in pixels (needed if gt_normalized=False).
        normalize_range: Target normalization range (default 1000).

    Returns:
        The annotation text, with bboxes converted to xyxy normalized form.
    """
    text = path.read_text(encoding="utf-8")

    if gt_format == "xyxy" and gt_normalized:
        # Already in the right format
        return text

    # Need to transform bboxes — parse line by line
    import re

    bbox_pattern = re.compile(r"\[([^\]]+)\]")
    lines = text.split("\n")
    out_lines = []

    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue

        match = bbox_pattern.search(line)
        if not match:
            out_lines.append(line)
            continue

        bbox_str = match.group(1)
        try:
            coords = [int(float(x.strip())) for x in bbox_str.split(",") if x.strip()]
            if len(coords) != 4:
                out_lines.append(line)
                continue
        except ValueError:
            out_lines.append(line)
            continue

        # Convert xywh -> xyxy if needed
        if gt_format == "xywh":
            coords = xywh_to_xyxy(coords)

        # Normalize if needed
        if not gt_normalized and img_width and img_height:
            coords = normalize_bbox(coords, img_width, img_height, normalize_range)

        new_bbox_str = f"[{coords[0]},{coords[1]},{coords[2]},{coords[3]}]"
        line = line[: match.start()] + new_bbox_str + line[match.end() :]
        out_lines.append(line)

    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Metadata loading (for image dimensions)
# ---------------------------------------------------------------------------


def load_metadata(meta_dir: Path, stem: str) -> Optional[Dict]:
    """Load metadata JSON for a sample (for image dimensions)."""
    meta_path = meta_dir / f"{stem}.json"
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Per-sample worker (module-level so it's picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _evaluate_one(job: Dict) -> Dict:
    """Evaluate a single sample. Runs inside a worker process."""
    stem = job["stem"]
    gt_path = Path(job["gt_path"])
    pred_path = Path(job["pred_path"])

    try:
        img_w, img_h = None, None
        if not job["gt_normalized"] and job["meta_dir"]:
            meta = load_metadata(Path(job["meta_dir"]), stem)
            if meta:
                img_w = meta.get("image_width")
                img_h = meta.get("image_height")

        gt_text = load_annotation(
            gt_path,
            gt_format=job["gt_format"],
            gt_normalized=job["gt_normalized"],
            img_width=img_w,
            img_height=img_h,
            normalize_range=config.NORMALIZE_RANGE,
        )
        pred_text = pred_path.read_text(encoding="utf-8")

        role_mapping = config.SIMPLIFIED_ROLE_MAPPING if job["simplified_roles"] else None

        result = evaluate_sample(
            gt_text,
            pred_text,
            iou_threshold=job["iou_threshold"],
            compute_ged_flag=job["compute_ged"],
            ged_timeout=job["ged_timeout"],
            ged_max_nodes=job["ged_max_nodes"],
            role_mapping=role_mapping,
        )

        gt_nodes = parse_tree(gt_text, role_mapping=role_mapping)
        pred_nodes = parse_tree(pred_text, role_mapping=role_mapping)
        result["gt_roles"] = [n.role for n in gt_nodes]
        result["pred_roles"] = [n.role for n in pred_nodes]
        result["filename"] = stem
        return result

    except Exception as e:
        return {
            "filename": stem,
            "parsed": False,
            "edge_f1": 0.0,
            "leaves_f1": 0.0,
            "container_match": 0.0,
            "ged": None,
            "gt_node_count": 0,
            "pred_node_count": 0,
            "matched_nodes": 0,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_evaluation(args: argparse.Namespace) -> None:
    """Run evaluation on predictions vs GT."""
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    if not gt_dir.exists():
        logger.error(f"GT directory not found: {gt_dir}")
        return
    if not pred_dir.exists():
        logger.error(f"Predictions directory not found: {pred_dir}")
        return

    # Discover GT files
    gt_files = sorted(gt_dir.glob("*.txt"))
    if not gt_files:
        logger.error(f"No .txt files found in {gt_dir}")
        return

    logger.info(f"Found {len(gt_files)} GT annotations in {gt_dir}")

    # Limit samples if requested
    if args.num_samples is not None:
        gt_files = gt_files[: args.num_samples]

    # Optional metadata dir (for image dimensions when gt_normalized=False)
    meta_dir = None
    if args.metadata_dir:
        meta_dir = Path(args.metadata_dir)

    # Build job list
    jobs: List[Dict] = []
    n_missing = 0
    for gt_path in gt_files:
        stem = gt_path.stem
        pred_path = pred_dir / f"{stem}.txt"
        if not pred_path.exists():
            logger.warning(f"No prediction found for {stem}")
            n_missing += 1
            continue
        jobs.append({
            "stem": stem,
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
            "gt_format": args.gt_format,
            "gt_normalized": args.gt_normalized,
            "meta_dir": str(meta_dir) if meta_dir else None,
            "iou_threshold": args.iou_threshold,
            "compute_ged": not args.skip_ged,
            "ged_timeout": args.ged_timeout,
            "ged_max_nodes": args.ged_max_nodes,
            "simplified_roles": args.simplified_roles,
        })

    # Process samples (parallel if workers > 1)
    per_sample_results: List[Dict] = []
    if args.workers and args.workers > 1:
        logger.info(f"Evaluating with {args.workers} worker processes")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_evaluate_one, j): j["stem"] for j in jobs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                per_sample_results.append(fut.result())
        # Results arrive out of order — sort by filename for deterministic output
        def _sort_key(r: Dict):
            try:
                return (0, int(r.get("filename", "")))
            except (TypeError, ValueError):
                return (1, r.get("filename", ""))
        per_sample_results.sort(key=_sort_key)
    else:
        for job in tqdm(jobs, desc="Evaluating"):
            per_sample_results.append(_evaluate_one(job))

    if not per_sample_results:
        logger.error("No samples evaluated.")
        return

    # Aggregate
    summary = evaluate_batch(per_sample_results)
    summary["model"] = args.model or config.MODEL_NAME

    # Print results
    print(f"\n{'='*45}")
    print(f" Screen2AX Evaluation Results")
    print(f"{'='*45}")
    print(f"Model: {summary['model']}")
    print(
        f"Samples: {summary['n_total']} total, {summary['n_parsed']} parsed "
        f"({summary['parse_rate']*100:.1f}% parse rate)"
    )
    if n_missing:
        print(f"Missing predictions: {n_missing}")
    print()

    header = f"{'Metric':<20} {'Mean':>8} {'Std':>8} {'Median':>8}"
    print(header)
    print("\u2500" * len(header))

    for metric_name, key in [
        ("Edge F1", "edge_f1"),
        ("Leaves F1", "leaves_f1"),
        ("Container Match", "container_match"),
    ]:
        s = summary[key]
        print(f"{metric_name:<20} {s['mean']:>8.3f} {s['std']:>8.3f} {s['median']:>8.3f}")

    if not args.skip_ged:
        s = summary["ged"]
        print(f"{'GED':<20} {s['mean']:>8.1f} {s['std']:>8.1f} {s['median']:>8.1f}")
        print(f"  (computed for {summary['n_ged_computed']}/{summary['n_parsed']} samples)")

    # Micro-averaged metrics (pooled counts across the dataset)
    print()
    header2 = f"{'Metric (micro)':<20} {'Prec':>8} {'Recall':>8} {'F1':>8}"
    print(header2)
    print("\u2500" * len(header2))
    for metric_name, key in [
        ("Edge F1 (micro)", "edge_f1_micro"),
        ("Leaves F1 (micro)", "leaves_f1_micro"),
    ]:
        m = summary[key]
        print(f"{metric_name:<20} {m['precision']:>8.3f} {m['recall']:>8.3f} {m['f1']:>8.3f}")

    if not args.skip_ged:
        print(f"{'GED (sum)':<20} {summary['ged_total']:>8.1f}")

    print()
    print(f"Avg GT nodes:       {summary['avg_gt_nodes']:.1f}")
    print(f"Avg Pred nodes:     {summary['avg_pred_nodes']:.1f}")

    # Save aggregated metrics
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove non-serializable fields from summary
        save_summary = {k: v for k, v in summary.items()}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics saved to {output_path}")

    # Save per-sample results
    if args.per_sample and args.output:
        output_path = Path(args.output)
        per_sample_path = output_path.parent / (output_path.stem + "_per_sample.jsonl")
        with open(per_sample_path, "w", encoding="utf-8") as f:
            for r in per_sample_results:
                # Remove lists from per-sample output to keep it compact
                row = {
                    k: v
                    for k, v in r.items()
                    if k not in ("gt_roles", "pred_roles")
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"Per-sample metrics saved to {per_sample_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Screen2AX metrics on saved predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        required=True,
        help="Directory with GT annotation .txt files",
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Directory with predicted .txt files (filenames must match GT)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save aggregated metrics JSON",
    )
    parser.add_argument(
        "--per-sample",
        action="store_true",
        help="Also save per-sample metrics to {output_stem}_per_sample.jsonl",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Evaluate only first N samples",
    )
    parser.add_argument(
        "--skip-ged",
        action="store_true",
        help="Skip GED computation (much faster)",
    )
    parser.add_argument(
        "--ged-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for GED per sample",
    )
    parser.add_argument(
        "--ged-max-nodes",
        type=int,
        default=100,
        help="Skip GED when either tree has more than this many nodes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel evaluation (1 = sequential)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for node matching",
    )
    parser.add_argument(
        "--gt-format",
        type=str,
        choices=["xyxy", "xywh"],
        default="xyxy",
        help="Bounding box format in GT annotations",
    )
    parser.add_argument(
        "--gt-normalized",
        action="store_true",
        default=True,
        help="GT coords are already normalized to 0-1000 (default True)",
    )
    parser.add_argument(
        "--no-gt-normalized",
        action="store_true",
        help="GT coords are in pixel space (need image dimensions to normalize)",
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default=None,
        help="Directory with metadata JSONs (for image dimensions when GT is not normalized)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for reporting (default: from config)",
    )
    parser.add_argument(
        "--simplified-roles",
        action="store_true",
        help="Use 7-class simplified role mapping from Screen2AX paper",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Handle --no-gt-normalized flag
    if args.no_gt_normalized:
        args.gt_normalized = False

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    run_evaluation(args)
