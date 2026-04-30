"""Screen2AX evaluation metrics.

Implements:
1. Edge F1 -- decompose tree into parent-child edges, compute F1
2. Leaves F1 -- F1 only on edges where child is a leaf
3. Complete Match (CM) -- 1.0 if all edges match exactly, else 0.0
4. Graph Edit Distance (GED) -- via networkx optimize_graph_edit_distance
"""

from __future__ import annotations

import logging
import math
import signal
import threading
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .parser import AXNode, parse_tree, is_leaf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounding-box IoU
# ---------------------------------------------------------------------------

def compute_iou(box_a: List[int], box_b: List[int]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


# ---------------------------------------------------------------------------
# Node matching via Hungarian algorithm
# ---------------------------------------------------------------------------

def match_nodes(
    gt_nodes: List[AXNode],
    pred_nodes: List[AXNode],
    iou_threshold: float = 0.5,
) -> Dict[int, int]:
    """Match predicted nodes to GT nodes using Hungarian algorithm.

    Returns a dict mapping pred node_id -> gt node_id for valid matches.
    A match is valid only if roles match exactly and IoU >= threshold.
    """
    if not gt_nodes or not pred_nodes:
        return {}

    n_gt = len(gt_nodes)
    n_pred = len(pred_nodes)
    large_cost = 1e6

    # Build cost matrix: rows = pred, cols = gt
    cost_matrix = np.full((n_pred, n_gt), large_cost)

    for i, pred in enumerate(pred_nodes):
        for j, gt in enumerate(gt_nodes):
            # Roles must match exactly
            if pred.role != gt.role:
                continue
            # Both must have bboxes
            if pred.bbox is None or gt.bbox is None:
                continue
            iou = compute_iou(pred.bbox, gt.bbox)
            if iou >= iou_threshold:
                cost_matrix[i, j] = 1.0 - iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping: Dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < large_cost:
            mapping[pred_nodes[r].node_id] = gt_nodes[c].node_id

    return mapping


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------

def extract_edges(nodes: List[AXNode]) -> Set[Tuple[int, int]]:
    """Extract all parent-child edges from a node list as (parent_id, child_id) tuples."""
    edges: Set[Tuple[int, int]] = set()
    for node in nodes:
        for child in node.children:
            edges.add((node.node_id, child.node_id))
    return edges


def extract_leaf_edges(nodes: List[AXNode]) -> Set[Tuple[int, int]]:
    """Extract edges where the child is a leaf node."""
    edges: Set[Tuple[int, int]] = set()
    for node in nodes:
        for child in node.children:
            if is_leaf(child):
                edges.add((node.node_id, child.node_id))
    return edges


# ---------------------------------------------------------------------------
# Edge remapping
# ---------------------------------------------------------------------------

def remap_edges(
    edges: Set[Tuple[int, int]],
    mapping: Dict[int, int],
) -> Set[Tuple[int, int]]:
    """Remap predicted edges to GT node ID space using the node matching.

    Only edges where both parent and child have a GT match are included.
    """
    remapped: Set[Tuple[int, int]] = set()
    for parent_id, child_id in edges:
        if parent_id in mapping and child_id in mapping:
            remapped.add((mapping[parent_id], mapping[child_id]))
    return remapped


# ---------------------------------------------------------------------------
# F1 computation
# ---------------------------------------------------------------------------

def compute_f1(gt_set: Set, pred_set: Set) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 between two sets."""
    if not gt_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0
    if not gt_set:
        return 0.0, 0.0, 0.0

    tp = len(gt_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0

    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Graph Edit Distance
# ---------------------------------------------------------------------------

def _build_digraph(nodes: List[AXNode]):
    """Build a networkx DiGraph from AXNodes."""
    import networkx as nx

    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node.node_id, role=node.role)
    for node in nodes:
        for child in node.children:
            G.add_edge(node.node_id, child.node_id)
    return G


def compute_ged(
    gt_nodes: List[AXNode],
    pred_nodes: List[AXNode],
    timeout: float = 30.0,
    max_nodes: int = 100,
) -> Optional[float]:
    """Compute Graph Edit Distance between GT and predicted trees.

    Uses networkx optimize_graph_edit_distance with cost functions:
    - node_subst_cost: 0 if roles match, 1 otherwise
    - node_del_cost / node_ins_cost: 1
    - edge_subst_cost: 0  (edges have no attributes)
    - edge_del_cost / edge_ins_cost: 1

    Returns None if skipped (too many nodes) or timed out.
    """
    import networkx as nx

    if len(gt_nodes) > max_nodes or len(pred_nodes) > max_nodes:
        logger.info(
            f"Skipping GED: gt={len(gt_nodes)}, pred={len(pred_nodes)} "
            f"(max={max_nodes})"
        )
        return None

    G_gt = _build_digraph(gt_nodes)
    G_pred = _build_digraph(pred_nodes)

    def node_subst_cost(u_attrs, v_attrs):
        return 0.0 if u_attrs.get("role") == v_attrs.get("role") else 1.0

    def node_del_cost(u_attrs):
        return 1.0

    def node_ins_cost(v_attrs):
        return 1.0

    def edge_subst_cost(e1_attrs, e2_attrs):
        return 0.0

    def edge_del_cost(e_attrs):
        return 1.0

    def edge_ins_cost(e_attrs):
        return 1.0

    # networkx's optimize_graph_edit_distance is a generator, but it can spend
    # a long time computing each yield on non-trivial graphs — a timeout check
    # between iterations is useless if the first iteration never completes.
    # Run it in a background daemon thread; the main thread enforces the
    # wall-clock cutoff and returns the best value seen so far.
    best_ged: Optional[float] = None
    best_lock = threading.Lock()
    done = threading.Event()

    def _worker() -> None:
        nonlocal best_ged
        try:
            gen = nx.optimize_graph_edit_distance(
                G_gt,
                G_pred,
                node_subst_cost=node_subst_cost,
                node_del_cost=node_del_cost,
                node_ins_cost=node_ins_cost,
                edge_subst_cost=edge_subst_cost,
                edge_del_cost=edge_del_cost,
                edge_ins_cost=edge_ins_cost,
            )
            for ged_value in gen:
                with best_lock:
                    best_ged = ged_value
                if done.is_set():
                    break
        except Exception as e:
            logger.warning(f"GED computation failed: {e}")
        finally:
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    finished = done.wait(timeout=timeout)
    if not finished:
        done.set()  # signal worker to stop at next yield (best-effort)
        logger.debug(
            f"GED timeout after {timeout}s, best={best_ged} "
            f"(worker left running in background)"
        )

    with best_lock:
        if best_ged is not None:
            return best_ged

    # Timeout with no yielded value: approximate GED as the number of edges
    # in the GT tree (matches the Screen2AX paper's fallback).
    gt_edge_count = float(sum(len(n.children) for n in gt_nodes))
    logger.debug(
        f"GED timed out with no value; falling back to GT edge count = {gt_edge_count}"
    )
    return gt_edge_count


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(
    gt_text: str,
    pred_text: str,
    iou_threshold: float = 0.5,
    compute_ged_flag: bool = True,
    ged_timeout: float = 30.0,
    ged_max_nodes: int = 100,
    role_mapping: Optional[Dict[str, Optional[str]]] = None,
) -> Dict:
    """Evaluate a single sample: returns dict with all metrics.

    In addition to per-sample F1 scores, the result carries raw TP/FP/FN
    counts (``edge_tp``, ``edge_fp``, ``edge_fn``, and the leaves_* variants)
    so that :func:`evaluate_batch` can also compute micro-averaged F1.
    """
    result: Dict = {
        "parsed": False,
        "edge_f1": 0.0,
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "edge_tp": 0,
        "edge_fp": 0,
        "edge_fn": 0,
        "leaves_f1": 0.0,
        "leaves_precision": 0.0,
        "leaves_recall": 0.0,
        "leaves_tp": 0,
        "leaves_fp": 0,
        "leaves_fn": 0,
        "container_match": 0.0,
        "n_containers": 0,
        "ged": None,
        "gt_node_count": 0,
        "pred_node_count": 0,
        "matched_nodes": 0,
    }

    # Parse trees
    gt_nodes = parse_tree(gt_text, role_mapping=role_mapping)
    pred_nodes = parse_tree(pred_text, role_mapping=role_mapping)

    result["gt_node_count"] = len(gt_nodes)
    result["pred_node_count"] = len(pred_nodes)

    if not pred_nodes:
        return result

    result["parsed"] = True

    # Node matching
    mapping = match_nodes(gt_nodes, pred_nodes, iou_threshold=iou_threshold)
    result["matched_nodes"] = len(mapping)

    # Edge F1
    gt_edges = extract_edges(gt_nodes)
    pred_edges = extract_edges(pred_nodes)
    remapped_pred_edges = remap_edges(pred_edges, mapping)

    prec, rec, f1 = compute_f1(gt_edges, remapped_pred_edges)
    result["edge_precision"] = prec
    result["edge_recall"] = rec
    result["edge_f1"] = f1
    edge_tp = len(gt_edges & remapped_pred_edges)
    result["edge_tp"] = edge_tp
    result["edge_fp"] = len(remapped_pred_edges) - edge_tp
    result["edge_fn"] = len(gt_edges) - edge_tp

    # Leaves F1
    gt_leaf_edges = extract_leaf_edges(gt_nodes)
    pred_leaf_edges = extract_leaf_edges(pred_nodes)
    remapped_pred_leaf_edges = remap_edges(pred_leaf_edges, mapping)

    lprec, lrec, lf1 = compute_f1(gt_leaf_edges, remapped_pred_leaf_edges)
    result["leaves_precision"] = lprec
    result["leaves_recall"] = lrec
    result["leaves_f1"] = lf1
    leaves_tp = len(gt_leaf_edges & remapped_pred_leaf_edges)
    result["leaves_tp"] = leaves_tp
    result["leaves_fp"] = len(remapped_pred_leaf_edges) - leaves_tp
    result["leaves_fn"] = len(gt_leaf_edges) - leaves_tp

    # Container Match (paper's "CM"): mean IoU over matched intermediate
    # (non-leaf) GT nodes. 0 if the tree has no matched intermediate nodes.
    gt_by_id = {n.node_id: n for n in gt_nodes}
    pred_by_id = {n.node_id: n for n in pred_nodes}
    gt_to_pred = {gt_id: pred_id for pred_id, gt_id in mapping.items()}

    container_ious: List[float] = []
    for gt in gt_nodes:
        if is_leaf(gt) or gt.bbox is None:
            continue
        pred_id = gt_to_pred.get(gt.node_id)
        if pred_id is None:
            container_ious.append(0.0)
            continue
        pred = pred_by_id.get(pred_id)
        if pred is None or pred.bbox is None:
            container_ious.append(0.0)
            continue
        container_ious.append(compute_iou(gt.bbox, pred.bbox))

    result["container_match"] = (
        float(np.mean(container_ious)) if container_ious else 0.0
    )
    result["n_containers"] = len(container_ious)

    # Graph Edit Distance
    if compute_ged_flag:
        result["ged"] = compute_ged(
            gt_nodes,
            pred_nodes,
            timeout=ged_timeout,
            max_nodes=ged_max_nodes,
        )

    return result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def _safe_stats(values: List[float]) -> Dict:
    """Compute mean, std, median for a list of floats."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
    }


def _micro_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute micro-averaged precision/recall/F1 from pooled counts."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def evaluate_batch(results: List[Dict]) -> Dict:
    """Aggregate per-sample metrics into summary statistics.

    Reports both macro-averaged stats (mean/std/median across samples) and
    micro-averaged F1 / summed GED, which pool TP/FP/FN counts across the
    whole dataset before computing the metric.
    """
    n_total = len(results)
    n_parsed = sum(1 for r in results if r.get("parsed", False))
    parse_rate = n_parsed / n_total if n_total > 0 else 0.0

    edge_f1s = [r["edge_f1"] for r in results if r.get("parsed")]
    leaves_f1s = [r["leaves_f1"] for r in results if r.get("parsed")]
    cms = [r["container_match"] for r in results if r.get("parsed")]
    geds = [r["ged"] for r in results if r.get("parsed") and r.get("ged") is not None]

    # Micro-averaged counts (pooled across all parsed samples)
    edge_tp = sum(r.get("edge_tp", 0) for r in results if r.get("parsed"))
    edge_fp = sum(r.get("edge_fp", 0) for r in results if r.get("parsed"))
    edge_fn = sum(r.get("edge_fn", 0) for r in results if r.get("parsed"))
    leaves_tp = sum(r.get("leaves_tp", 0) for r in results if r.get("parsed"))
    leaves_fp = sum(r.get("leaves_fp", 0) for r in results if r.get("parsed"))
    leaves_fn = sum(r.get("leaves_fn", 0) for r in results if r.get("parsed"))

    gt_counts = [r["gt_node_count"] for r in results]
    pred_counts = [r["pred_node_count"] for r in results if r.get("parsed")]

    # Role distribution
    gt_role_counter: Counter = Counter()
    pred_role_counter: Counter = Counter()
    for r in results:
        if "gt_roles" in r:
            gt_role_counter.update(r["gt_roles"])
        if "pred_roles" in r and r.get("parsed"):
            pred_role_counter.update(r["pred_roles"])

    gt_total_roles = sum(gt_role_counter.values()) or 1
    pred_total_roles = sum(pred_role_counter.values()) or 1

    return {
        "n_total": n_total,
        "n_parsed": n_parsed,
        "parse_rate": round(parse_rate, 4),
        "edge_f1": _safe_stats(edge_f1s),
        "leaves_f1": _safe_stats(leaves_f1s),
        "container_match": _safe_stats(cms),
        "ged": _safe_stats(geds),
        "edge_f1_micro": _micro_f1(edge_tp, edge_fp, edge_fn),
        "leaves_f1_micro": _micro_f1(leaves_tp, leaves_fp, leaves_fn),
        "ged_total": float(sum(geds)) if geds else 0.0,
        "n_ged_computed": len(geds),
        "avg_gt_nodes": round(np.mean(gt_counts), 1) if gt_counts else 0.0,
        "avg_pred_nodes": round(np.mean(pred_counts), 1) if pred_counts else 0.0,
        "role_distribution_gt": {
            role: round(count / gt_total_roles, 4)
            for role, count in gt_role_counter.most_common(30)
        },
        "role_distribution_pred": {
            role: round(count / pred_total_roles, 4)
            for role, count in pred_role_counter.most_common(30)
        },
    }
