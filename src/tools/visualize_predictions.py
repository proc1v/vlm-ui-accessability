#!/usr/bin/env python3
"""
Visualize ShowUI zero-shot predictions vs Screen2AX-Element ground truth.

Reads the JSON output from run_showui_zeroshot.py and generates:
  - Ground truth bounding box overlay
  - Strategy A prediction overlay
  - Strategy C grounding point overlay
  - Side-by-side GT vs prediction comparison
  - Summary report (text)
"""

import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "AXButton": "#e6194b",
    "AXDisclosureTriangle": "#3cb44b",
    "AXImage": "#4363d8",
    "AXLink": "#f58231",
    "AXTextArea": "#911eb4",
}

FALLBACK_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                   "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990"]

DPI = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def color_for_category(name):
    return CATEGORY_COLORS.get(name, FALLBACK_COLORS[hash(name) % len(FALLBACK_COLORS)])


def draw_bbox(ax, bbox_xyxy, color, label=None, linewidth=2, alpha=0.8):
    """Draw a bounding box rectangle on a matplotlib axes."""
    x1, y1, x2, y2 = bbox_xyxy
    rect = mpatches.FancyBboxPatch(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=linewidth, edgecolor=color, facecolor="none",
        alpha=alpha, boxstyle="square,pad=0",
    )
    ax.add_patch(rect)
    if label:
        ax.text(x1, max(y1 - 4, 0), label, fontsize=5, color="white",
                bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"),
                verticalalignment="bottom")


def coco_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def load_image(image_id, predictions_dir):
    """Load the saved screenshot for a given image_id."""
    path = os.path.join(predictions_dir, f"{image_id}.png")
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def fig_ground_truth(img, gt_elements, image_id):
    """Figure 1: Ground truth bounding boxes."""
    fig, ax = plt.subplots(1, 1, figsize=(img.width / DPI * 1.5, img.height / DPI * 1.5), dpi=DPI)
    ax.imshow(img)
    ax.set_axis_off()

    counts = {}
    for el in gt_elements:
        cat = el["category_name"]
        counts[cat] = counts.get(cat, 0) + 1
        bbox = el.get("bbox_xyxy") or coco_to_xyxy(el["bbox"])
        draw_bbox(ax, bbox, color_for_category(cat), label=cat)

    # Legend
    handles = [mpatches.Patch(facecolor=color_for_category(c), label=f"{c} ({n})")
               for c, n in sorted(counts.items())]
    ax.legend(handles=handles, loc="upper right", fontsize=6, framealpha=0.8)
    ax.set_title(f"Ground Truth — image {image_id} ({sum(counts.values())} elements)", fontsize=8)
    fig.tight_layout()
    return fig


def fig_strategy_a(img, strategy_a, image_id):
    """Figure 2: Strategy A predictions."""
    fig, ax = plt.subplots(1, 1, figsize=(img.width / DPI * 1.5, img.height / DPI * 1.5), dpi=DPI)
    ax.imshow(img)
    ax.set_axis_off()

    if strategy_a.get("parse_failed") or strategy_a.get("error"):
        raw = strategy_a.get("raw_output", strategy_a.get("error", "No output"))
        wrapped = textwrap.fill(str(raw)[:500], width=80)
        ax.text(
            0.05, 0.95, f"Parse failed. Raw output:\n{wrapped}",
            transform=ax.transAxes, fontsize=5, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="red"),
        )
        ax.set_title(f"Strategy A — image {image_id} (PARSE FAILED)", fontsize=8)
    else:
        elements = strategy_a.get("parsed_elements", [])
        for el in elements:
            if isinstance(el, dict) and "bbox" in el:
                etype = el.get("type", "unknown")
                # Attempt to map predicted type to our category colors
                color = "#888888"
                for cat, col in CATEGORY_COLORS.items():
                    if etype.lower() in cat.lower() or cat.lower().replace("ax", "") in etype.lower():
                        color = col
                        break
                draw_bbox(ax, el["bbox"], color, label=etype)
        ax.set_title(
            f"Strategy A — image {image_id} ({len(elements)} predicted elements)",
            fontsize=8,
        )
    fig.tight_layout()
    return fig


def fig_grounding(img, strategy_c, gt_elements, image_id, img_w, img_h):
    """Figure 3: Strategy C grounding points."""
    fig, ax = plt.subplots(1, 1, figsize=(img.width / DPI * 1.5, img.height / DPI * 1.5), dpi=DPI)
    ax.imshow(img)
    ax.set_axis_off()

    # Draw GT boxes as thin outlines
    for el in gt_elements:
        bbox = el.get("bbox_xyxy") or coco_to_xyxy(el["bbox"])
        draw_bbox(ax, bbox, "#aaaaaa", linewidth=1, alpha=0.5)

    queries = strategy_c.get("queries", [])
    for q in queries:
        pt = q.get("predicted_point")
        if pt and pt[0] is not None:
            px, py = pt[0] * img_w, pt[1] * img_h
            color = "#00cc00" if q["hit"] else "#ff0000"
            ax.plot(px, py, "o", markersize=6, markeredgewidth=1.5,
                    markeredgecolor="white", markerfacecolor=color, alpha=0.9)

    hit_rate = strategy_c.get("hit_rate", 0)
    hits = strategy_c.get("hits", 0)
    total = strategy_c.get("total", 0)
    ax.set_title(
        f"Strategy C Grounding — image {image_id} "
        f"(hit rate: {hit_rate:.1%}, {hits}/{total})",
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def fig_comparison(img, gt_elements, strategy_a, image_id):
    """Figure 4: Side-by-side GT vs Strategy A."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(img.width / DPI * 3, img.height / DPI * 1.5), dpi=DPI)

    for ax in (ax1, ax2):
        ax.imshow(img)
        ax.set_axis_off()

    # Left: GT
    for el in gt_elements:
        bbox = el.get("bbox_xyxy") or coco_to_xyxy(el["bbox"])
        draw_bbox(ax1, bbox, color_for_category(el["category_name"]))
    ax1.set_title(f"Ground Truth ({len(gt_elements)} elements)", fontsize=8)

    # Right: Strategy A
    pred_elements = []
    if not strategy_a.get("parse_failed") and not strategy_a.get("error"):
        pred_elements = strategy_a.get("parsed_elements", [])
        for el in pred_elements:
            if isinstance(el, dict) and "bbox" in el:
                draw_bbox(ax2, el["bbox"], "#4363d8")
    n_pred = len(pred_elements)
    ax2.set_title(f"Strategy A Predictions ({n_pred} elements)", fontsize=8)

    fig.suptitle(
        f"Image {image_id}: GT {len(gt_elements)} elements | Predicted {n_pred} elements",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_summary(samples, output_dir):
    """Generate summary_report.txt."""
    lines = ["ShowUI Zero-Shot Evaluation — Summary Report", "=" * 50, ""]

    total_hit_rates = []
    total_gt_counts = []
    total_pred_a_counts = []

    for s in samples:
        img_id = s["image_id"]
        gt_count = len(s["ground_truth"])
        total_gt_counts.append(gt_count)

        preds = s.get("predictions", {})
        lines.append(f"Image {img_id} ({s['image_size'][0]}x{s['image_size'][1]})")
        lines.append(f"  GT elements: {gt_count}")

        # Strategy A
        sa = preds.get("strategy_a", {})
        if sa.get("error"):
            lines.append(f"  Strategy A: ERROR — {sa['error']}")
            total_pred_a_counts.append(0)
        elif sa.get("parse_failed"):
            lines.append(f"  Strategy A: PARSE FAILED")
            total_pred_a_counts.append(0)
        else:
            n = len(sa.get("parsed_elements", []))
            total_pred_a_counts.append(n)
            lines.append(f"  Strategy A: {n} predicted elements ({sa.get('inference_time_s', '?')}s)")

        # Strategy B
        sb = preds.get("strategy_b", {})
        if sb.get("error"):
            lines.append(f"  Strategy B: ERROR — {sb['error']}")
        else:
            per_class = sb.get("per_class", {})
            for cls, data in per_class.items():
                n = len(data.get("parsed_elements", [])) if not data.get("parse_failed") else "PARSE_FAILED"
                lines.append(f"    {cls}: {n}")

        # Strategy C
        sc = preds.get("strategy_c", {})
        if sc.get("error"):
            lines.append(f"  Strategy C: ERROR — {sc['error']}")
        else:
            hr = sc.get("hit_rate", 0)
            total_hit_rates.append(hr)
            lines.append(f"  Strategy C: hit_rate={hr:.1%} ({sc.get('hits', 0)}/{sc.get('total', 0)})")

        lines.append("")

    # Aggregate
    lines.append("=" * 50)
    lines.append("AGGREGATE")
    lines.append(f"  Samples: {len(samples)}")
    if total_hit_rates:
        lines.append(f"  Average Strategy C hit rate: {np.mean(total_hit_rates):.1%}")
    if total_gt_counts and total_pred_a_counts:
        avg_gt = np.mean(total_gt_counts)
        avg_pred = np.mean(total_pred_a_counts)
        ratio = avg_pred / avg_gt if avg_gt > 0 else 0
        lines.append(f"  Avg GT elements/image: {avg_gt:.1f}")
        lines.append(f"  Avg Strategy A predictions/image: {avg_pred:.1f}")
        lines.append(f"  Predicted/GT ratio: {ratio:.2f}")
    lines.append("")

    # Qualitative notes
    lines.append("QUALITATIVE NOTES")
    lines.append("  (Automatically generated based on Strategy B per-class counts)")
    class_totals = {}
    for s in samples:
        sb = s.get("predictions", {}).get("strategy_b", {})
        for cls, data in sb.get("per_class", {}).items():
            if not data.get("parse_failed") and not data.get("error"):
                class_totals[cls] = class_totals.get(cls, 0) + len(data.get("parsed_elements", []))

    if class_totals:
        sorted_cls = sorted(class_totals.items(), key=lambda x: x[1], reverse=True)
        lines.append(f"  Most detected:  {sorted_cls[0][0]} ({sorted_cls[0][1]} total)")
        lines.append(f"  Least detected: {sorted_cls[-1][0]} ({sorted_cls[-1][1]} total)")

    report = "\n".join(lines)
    path = os.path.join(output_dir, "summary_report.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"Summary report saved to {path}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize ShowUI predictions vs ground truth")
    parser.add_argument("--predictions_file", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Output directory")
    parser.add_argument("--sample_idx", type=int, default=None, help="Visualize only this sample index")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.predictions_file) as f:
        data = json.load(f)

    predictions_dir = os.path.dirname(args.predictions_file)
    samples = data["samples"]

    if args.sample_idx is not None:
        samples = [samples[args.sample_idx]]

    for i, sample in enumerate(samples):
        image_id = sample["image_id"]
        img = load_image(image_id, predictions_dir)
        if img is None:
            print(f"[{i + 1}] Skipping image_id={image_id}: screenshot not found in {predictions_dir}")
            continue

        img_w, img_h = sample["image_size"]
        gt = sample["ground_truth"]
        preds = sample.get("predictions", {})
        print(f"[{i + 1}/{len(samples)}] Generating visualizations for image_id={image_id}...")

        # Figure 1: Ground Truth
        fig = fig_ground_truth(img, gt, image_id)
        fig.savefig(os.path.join(args.output_dir, f"{image_id}_gt.png"), dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        # Figure 2: Strategy A
        sa = preds.get("strategy_a", {})
        fig = fig_strategy_a(img, sa, image_id)
        fig.savefig(os.path.join(args.output_dir, f"{image_id}_pred_a.png"), dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        # Figure 3: Strategy C grounding
        sc = preds.get("strategy_c", {})
        if sc and not sc.get("error"):
            fig = fig_grounding(img, sc, gt, image_id, img_w, img_h)
            fig.savefig(os.path.join(args.output_dir, f"{image_id}_grounding.png"), dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        # Figure 4: Comparison
        fig = fig_comparison(img, gt, sa, image_id)
        fig.savefig(os.path.join(args.output_dir, f"{image_id}_comparison.png"), dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # Summary report
    generate_summary(data["samples"], args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
