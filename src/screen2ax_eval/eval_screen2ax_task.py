#!/usr/bin/env python3
"""Screen2AX-Task evaluation: perception → GPT-5 selector → IoU success rate.

Pipeline (per sample):
  1. Load Stage A perception output (linearized AX tree OR Screen2AX JSON).
  2. Convert to a hierarchical Screen2AX-style JSON with depth-first integer IDs.
  3. Send {command, accessibility_json} to selector LLM (paper prompt).
  4. Parse integer ID → look up node → compute IoU vs. ground-truth box.
  5. Categorize: success / parse_error / perception_error / selection_error.

Outputs per run:
  results.csv             — per-sample rows
  metrics.json            — aggregated success rate + error breakdown
  prompts/{id}.json       — debug snapshots (only with --save-prompts)

Usage:
  python eval_screen2ax_task.py \\
    --pred-dir /path/to/parsed \\
    --pred-format linearized_ax \\
    --pred-coords normalized \\
    --gt-dir /workspace/data/screen2ax_task/annotations \\
    --output-dir /path/to/task_eval \\
    --selector-model gpt-5 \\
    --concurrency 8 \\
    --num-samples 20    # smoke test; omit for full run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.parser import AXNode, parse_tree, get_root_nodes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Selector prompt (Screen2AX paper, Appendix)
# ---------------------------------------------------------------------------

SELECTOR_PROMPT_TEMPLATE = """You are given a list of UI elements in JSON format, each with a unique numeric ID and accessibility attributes.
Your task is to determine which UI element should be clicked to perform a specific action.
Return only the numeric ID of the element that corresponds to the action.
Do not explain or output anything else.

Accessibility JSON: {accessibility_json}
Action: {action}
Which element should be clicked?"""


# Container roles — used only for the optional "drop groups" view.
CONTAINER_ROLES = {
    "AXGroup", "AXOpaqueProviderGroup", "AXRadioGroup", "AXSplitGroup",
    "AXTabGroup", "AXToolbar", "AXWebArea", "AXOutline", "AXBrowser",
    "AXPopover", "AXGrid", "AXList", "AXTable", "AXScrollArea",
    "AXWindow", "AXPage", "AXMenu", "AXMenuBar", "AXSheet", "AXSplitter",
    "AXGrowArea", "AXCell",
}


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """Unified hierarchical element regardless of source format."""
    node_id: int
    cls: str
    box: List[int]                  # [x1, y1, x2, y2] in pixel coords
    value: Optional[str] = None
    name: Optional[str] = None      # kept distinct from value (Screen2AX uses only `value`)
    desc: Optional[str] = None
    children: List["TreeNode"] = field(default_factory=list)


@dataclass
class SampleResult:
    sample_id: str
    command: str
    gt_box: List[float]
    image_w: int
    image_h: int
    pred_id_keep: Optional[int] = None
    pred_id_drop: Optional[int] = None
    pred_box_keep: Optional[List[int]] = None
    pred_box_drop: Optional[List[int]] = None
    iou_keep: float = 0.0
    iou_drop: float = 0.0
    outcome_keep: str = "parse_error"
    outcome_drop: str = "parse_error"
    n_elements_keep: int = 0
    n_elements_drop: int = 0
    selector_raw_keep: str = ""
    selector_raw_drop: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Format-specific loaders → unified TreeNode forest
# ---------------------------------------------------------------------------

def _denormalize_box(
    box: List[int], img_w: int, img_h: int, normalized: bool, scale: int = 1000
) -> List[int]:
    if not normalized:
        return [int(v) for v in box]
    x1, y1, x2, y2 = box
    return [
        int(round(x1 * img_w / scale)),
        int(round(y1 * img_h / scale)),
        int(round(x2 * img_w / scale)),
        int(round(y2 * img_h / scale)),
    ]


def load_linearized(
    text: str, img_w: int, img_h: int, normalized: bool
) -> List[TreeNode]:
    """Convert linearized AX tree text → forest of TreeNodes."""
    nodes = parse_tree(text)
    if not nodes:
        return []

    ax_to_tree: Dict[int, TreeNode] = {}
    for n in nodes:
        if n.bbox is None:
            continue
        box = _denormalize_box(n.bbox, img_w, img_h, normalized)
        cls = n.role
        ax_to_tree[id(n)] = TreeNode(
            node_id=-1,  # filled later by DFS renumber
            cls=cls,
            box=box,
            value=n.value,
            name=n.name,
            desc=n.desc,
        )

    # Wire children using parent's children list from parse_tree
    for n in nodes:
        if id(n) not in ax_to_tree:
            continue
        parent = ax_to_tree[id(n)]
        for c in n.children:
            if id(c) in ax_to_tree:
                parent.children.append(ax_to_tree[id(c)])

    roots = [ax_to_tree[id(n)] for n in get_root_nodes(nodes) if id(n) in ax_to_tree]
    return roots


def load_screen2ax_json(
    obj: Any, img_w: int, img_h: int, normalized: bool
) -> List[TreeNode]:
    """Convert Screen2AX JSON (single root dict OR list) → forest of TreeNodes."""
    def conv(d: Dict[str, Any]) -> Optional[TreeNode]:
        if not isinstance(d, dict):
            return None
        box = d.get("box")
        if not (isinstance(box, list) and len(box) == 4):
            return None
        cls = d.get("cls", "AXUnknown")
        # Screen2AX uses "Group" / "Text" — normalize to AX-prefixed roles
        if cls == "Group":
            cls = "AXGroup"
        elif cls == "Text":
            cls = "AXStaticText"
        node = TreeNode(
            node_id=-1,
            cls=cls,
            box=_denormalize_box(box, img_w, img_h, normalized),
            value=d.get("value"),
        )
        for c in d.get("children", []) or []:
            tc = conv(c)
            if tc is not None:
                node.children.append(tc)
        return node

    if isinstance(obj, list):
        roots = [conv(d) for d in obj]
        return [r for r in roots if r is not None]
    if isinstance(obj, dict):
        r = conv(obj)
        return [r] if r is not None else []
    return []


def assign_dfs_ids(roots: List[TreeNode]) -> List[TreeNode]:
    """Walk the forest depth-first, assigning sequential IDs. Returns flat list."""
    flat: List[TreeNode] = []

    def walk(n: TreeNode) -> None:
        n.node_id = len(flat)
        flat.append(n)
        for c in n.children:
            walk(c)

    for r in roots:
        walk(r)
    return flat


def serialize_tree(roots: List[TreeNode]) -> List[Dict[str, Any]]:
    """Tree → JSON-able list of dicts in Screen2AX shape."""
    def dump(n: TreeNode) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": n.node_id, "cls": n.cls, "box": n.box}
        if n.value:
            out["value"] = n.value
        if n.name:
            out["name"] = n.name
        if n.desc:
            out["desc"] = n.desc
        if n.children:
            out["children"] = [dump(c) for c in n.children]
        return out
    return [dump(r) for r in roots]


def filter_drop_groups(roots: List[TreeNode]) -> List[TreeNode]:
    """Return a forest with all container nodes dropped — leaves & non-container
    intermediates float up to root. Children of dropped groups become roots
    (i.e. preserved, not removed)."""
    new_flat: List[TreeNode] = []

    def collect(n: TreeNode) -> List[TreeNode]:
        recursed_children: List[TreeNode] = []
        for c in n.children:
            recursed_children.extend(collect(c))
        if n.cls in CONTAINER_ROLES:
            return recursed_children
        # rebuild this node with only non-container descendants kept directly
        clone = TreeNode(node_id=-1, cls=n.cls, box=list(n.box),
                         value=n.value, name=n.name, desc=n.desc,
                         children=recursed_children)
        return [clone]

    out: List[TreeNode] = []
    for r in roots:
        out.extend(collect(r))
    return out


# ---------------------------------------------------------------------------
# IoU + outcome categorization
# ---------------------------------------------------------------------------

def iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - inter
    return inter / union if union > 0 else 0.0


def categorize(
    flat: List[TreeNode], gt_box: List[float], pred_node: Optional[TreeNode],
    threshold: float = 0.5,
) -> Tuple[str, float]:
    if pred_node is None:
        # Selector failed — was perception sufficient?
        if any(iou(n.box, gt_box) >= threshold for n in flat):
            return "selection_error", 0.0
        return "parse_error", 0.0
    pred_iou = iou(pred_node.box, gt_box)
    if pred_iou >= threshold:
        return "success", pred_iou
    # Wrong pick — perception had something that would have worked?
    if any(iou(n.box, gt_box) >= threshold for n in flat):
        return "selection_error", pred_iou
    return "perception_error", pred_iou


# ---------------------------------------------------------------------------
# Selector LLM
# ---------------------------------------------------------------------------

def build_selector_prompt(roots: List[TreeNode], action: str) -> str:
    elements_json = json.dumps(serialize_tree(roots), ensure_ascii=False, separators=(",", ":"))
    return SELECTOR_PROMPT_TEMPLATE.format(accessibility_json=elements_json, action=action)


async def call_selector(
    client, model: str, prompt: str, max_completion_tokens: int = 2048,
    max_retries: int = 8, base_backoff: float = 2.0,
    reasoning_effort: Optional[str] = "minimal",
) -> Tuple[str, int, int]:
    """Single async OpenAI call with exponential backoff on 429/5xx.

    GPT-5 reserves completion tokens for internal reasoning. We pass
    reasoning_effort='minimal' (supported on gpt-5 family) and a generous
    token cap so the visible answer is never starved. Both fall back
    automatically for older models.

    Returns (raw_text, prompt_tokens, completion_tokens).
    """
    is_gpt5 = "gpt-5" in model.lower()
    base_kwargs: Dict[str, Any] = dict(
        model=model, messages=[{"role": "user", "content": prompt}],
    )
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            kwargs = dict(base_kwargs)
            if is_gpt5:
                kwargs["max_completion_tokens"] = max_completion_tokens
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
            else:
                kwargs["max_tokens"] = 32  # plenty for an integer
                kwargs["temperature"] = 0.0
            try:
                resp = await client.chat.completions.create(**kwargs)
            except TypeError:
                # SDK doesn't recognize one of the new params — strip & retry once.
                kwargs.pop("reasoning_effort", None)
                resp = await client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            return text, (usage.prompt_tokens if usage else 0), (usage.completion_tokens if usage else 0)
        except Exception as e:
            last_exc = e
            msg = str(e)
            transient = ("429" in msg or "rate limit" in msg.lower()
                         or "503" in msg or "502" in msg or "timeout" in msg.lower())
            if not transient or attempt == max_retries - 1:
                raise
            # Honour Retry-After if the SDK exception carries it
            wait = base_backoff * (2 ** attempt)
            ra = getattr(getattr(e, "response", None), "headers", {}) or {}
            try:
                ra_val = ra.get("retry-after") or ra.get("Retry-After")
                if ra_val is not None:
                    wait = max(wait, float(ra_val))
            except Exception:
                pass
            wait = min(wait, 60.0)
            logger.info(f"selector retry {attempt+1}/{max_retries} in {wait:.1f}s ({msg[:120]})")
            await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]


def parse_selector_response(raw: str, n_elements: int) -> Optional[int]:
    """Extract a valid 0..n-1 integer from the response, else None."""
    if not raw:
        return None
    # Take the first integer in the string
    import re
    m = re.search(r"-?\d+", raw)
    if not m:
        return None
    try:
        idx = int(m.group(0))
    except ValueError:
        return None
    if idx < 0 or idx >= n_elements:
        return None
    return idx


# ---------------------------------------------------------------------------
# Per-sample worker
# ---------------------------------------------------------------------------

def find_pred_file(pred_dir: Path, sample_id: str, fmt: str) -> Optional[Path]:
    if fmt == "linearized_ax":
        p = pred_dir / f"{sample_id}.txt"
        return p if p.exists() else None
    elif fmt == "screen2ax_json":
        p = pred_dir / f"{sample_id}.json"
        return p if p.exists() else None
    return None


def load_prediction_tree(
    pred_path: Path, fmt: str, img_w: int, img_h: int, normalized: bool,
) -> List[TreeNode]:
    if fmt == "linearized_ax":
        text = pred_path.read_text(encoding="utf-8", errors="ignore")
        return load_linearized(text, img_w, img_h, normalized)
    if fmt == "screen2ax_json":
        obj = json.loads(pred_path.read_text(encoding="utf-8", errors="ignore"))
        return load_screen2ax_json(obj, img_w, img_h, normalized)
    raise ValueError(fmt)


async def process_one(
    sample_id: str, gt: Dict[str, Any], pred_dir: Path, args, client, sem: asyncio.Semaphore,
) -> SampleResult:
    img_w, img_h = int(gt["image_width"]), int(gt["image_height"])
    gt_box = [float(v) for v in gt["gt_box"]]
    command = gt["command"]
    res = SampleResult(
        sample_id=sample_id, command=command, gt_box=gt_box,
        image_w=img_w, image_h=img_h,
    )

    pred_path = find_pred_file(pred_dir, sample_id, args.pred_format)
    if pred_path is None:
        res.error = "missing_prediction"
        return res

    try:
        roots = load_prediction_tree(pred_path, args.pred_format, img_w, img_h,
                                     normalized=(args.pred_coords == "normalized"))
    except Exception as e:
        res.error = f"load_failed: {e}"
        return res

    # Two views: keep groups vs. drop groups
    roots_keep = roots
    roots_drop = filter_drop_groups(roots)
    flat_keep = assign_dfs_ids(roots_keep)
    flat_drop = assign_dfs_ids(roots_drop)
    res.n_elements_keep = len(flat_keep)
    res.n_elements_drop = len(flat_drop)

    if not flat_keep and not flat_drop:
        res.error = "empty_tree"
        return res

    async def select_view(roots_, flat_) -> Tuple[Optional[int], Optional[TreeNode], str, int, int]:
        if not flat_:
            return None, None, "", 0, 0
        prompt = build_selector_prompt(roots_, command)
        t0 = time.time()
        try:
            raw, p_toks, c_toks = await call_selector(client, args.selector_model, prompt)
        except Exception as e:
            logger.warning(f"[{sample_id}] selector call failed: {e}")
            return None, None, str(e), 0, 0
        res.latency_s += time.time() - t0
        idx = parse_selector_response(raw, len(flat_))
        node = flat_[idx] if idx is not None else None
        return idx, node, raw, p_toks, c_toks

    # Run both views inside a single semaphore acquisition so a sample
    # completes (writes a row) before yielding to other queued samples.
    async with sem:
        idx_k, node_k, raw_k, p_k, c_k = await select_view(roots_keep, flat_keep)
        idx_d, node_d, raw_d, p_d, c_d = await select_view(roots_drop, flat_drop)

    res.pred_id_keep, res.pred_id_drop = idx_k, idx_d
    res.pred_box_keep = node_k.box if node_k else None
    res.pred_box_drop = node_d.box if node_d else None
    res.selector_raw_keep, res.selector_raw_drop = raw_k, raw_d
    res.prompt_tokens = p_k + p_d
    res.completion_tokens = c_k + c_d

    res.outcome_keep, res.iou_keep = categorize(flat_keep, gt_box, node_k)
    res.outcome_drop, res.iou_drop = categorize(flat_drop, gt_box, node_d)

    if args.save_prompts:
        snap_dir = Path(args.output_dir) / "prompts"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap = {
            "sample_id": sample_id,
            "command": command,
            "gt_box": gt_box,
            "keep": {"prompt": build_selector_prompt(roots_keep, command),
                     "response": raw_k, "picked_id": idx_k,
                     "picked_box": res.pred_box_keep, "iou": res.iou_keep,
                     "outcome": res.outcome_keep},
            "drop": {"prompt": build_selector_prompt(roots_drop, command),
                     "response": raw_d, "picked_id": idx_d,
                     "picked_box": res.pred_box_drop, "iou": res.iou_drop,
                     "outcome": res.outcome_drop},
        }
        (snap_dir / f"{sample_id}.json").write_text(
            json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "sample_id", "command", "image_w", "image_h", "gt_box",
    "n_elements_keep", "pred_id_keep", "pred_box_keep", "iou_keep", "outcome_keep", "selector_raw_keep",
    "n_elements_drop", "pred_id_drop", "pred_box_drop", "iou_drop", "outcome_drop", "selector_raw_drop",
    "prompt_tokens", "completion_tokens", "latency_s", "error",
]


def write_row(writer, r: SampleResult) -> None:
    writer.writerow({
        "sample_id": r.sample_id, "command": r.command,
        "image_w": r.image_w, "image_h": r.image_h,
        "gt_box": json.dumps(r.gt_box),
        "n_elements_keep": r.n_elements_keep, "pred_id_keep": r.pred_id_keep,
        "pred_box_keep": json.dumps(r.pred_box_keep) if r.pred_box_keep else "",
        "iou_keep": f"{r.iou_keep:.4f}", "outcome_keep": r.outcome_keep,
        "selector_raw_keep": r.selector_raw_keep[:200],
        "n_elements_drop": r.n_elements_drop, "pred_id_drop": r.pred_id_drop,
        "pred_box_drop": json.dumps(r.pred_box_drop) if r.pred_box_drop else "",
        "iou_drop": f"{r.iou_drop:.4f}", "outcome_drop": r.outcome_drop,
        "selector_raw_drop": r.selector_raw_drop[:200],
        "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
        "latency_s": f"{r.latency_s:.3f}", "error": r.error or "",
    })


def aggregate_metrics(rows: List[SampleResult], system_id: str, model: str) -> Dict[str, Any]:
    n = len(rows)
    def stats(view: str) -> Dict[str, Any]:
        outcomes = [getattr(r, f"outcome_{view}") for r in rows]
        ious = [getattr(r, f"iou_{view}") for r in rows]
        succ = sum(1 for o in outcomes if o == "success")
        return {
            "success_rate": succ / n if n else 0.0,
            "n_success": succ,
            "n_perception_error": sum(1 for o in outcomes if o == "perception_error"),
            "n_selection_error": sum(1 for o in outcomes if o == "selection_error"),
            "n_parse_error": sum(1 for o in outcomes if o == "parse_error"),
            "mean_iou": sum(ious) / n if n else 0.0,
            "mean_iou_on_successes":
                (sum(i for i, o in zip(ious, outcomes) if o == "success") / succ) if succ else 0.0,
        }
    return {
        "system_id": system_id,
        "selector_model": model,
        "n_samples": n,
        "keep_groups": stats("keep"),
        "drop_groups": stats("drop"),
        "total_prompt_tokens": sum(r.prompt_tokens for r in rows),
        "total_completion_tokens": sum(r.completion_tokens for r in rows),
        "n_load_errors": sum(1 for r in rows if r.error),
    }


def load_done_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["sample_id"])
    return done


async def run(args: argparse.Namespace) -> None:
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    metrics_path = out_dir / "metrics.json"

    # Discover samples by GT annotations (canonical source — pred may have gaps)
    sample_ids = sorted([p.stem for p in gt_dir.glob("*.json")],
                        key=lambda s: int(s) if s.isdigit() else s)
    if args.num_samples is not None:
        sample_ids = sample_ids[: args.num_samples]
    logger.info(f"{len(sample_ids)} candidate samples")

    done = load_done_ids(csv_path) if args.resume else set()
    todo = [s for s in sample_ids if s not in done]
    if args.resume and done:
        logger.info(f"Resuming: {len(done)} already done, {len(todo)} remaining")

    if not todo:
        logger.info("Nothing to do.")
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        sem = asyncio.Semaphore(args.concurrency)

        # Open CSV in append mode if resuming, else write fresh with header.
        write_header = not csv_path.exists() or not args.resume
        f = open(csv_path, "a" if args.resume and csv_path.exists() else "w",
                 newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
            f.flush()

        async def worker(sid: str) -> SampleResult:
            gt = json.loads((gt_dir / f"{sid}.json").read_text(encoding="utf-8"))
            return await process_one(sid, gt, pred_dir, args, client, sem)

        tasks = [asyncio.create_task(worker(sid)) for sid in todo]
        with tqdm(total=len(tasks), desc="eval") as pbar:
            for coro in asyncio.as_completed(tasks):
                r = await coro
                write_row(writer, r)
                f.flush()
                pbar.update(1)
        f.close()

    # Aggregate from full CSV (covers both newly written and resumed rows)
    rows: List[SampleResult] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = SampleResult(
                sample_id=row["sample_id"],
                command=row["command"],
                gt_box=json.loads(row["gt_box"]) if row["gt_box"] else [],
                image_w=int(row["image_w"] or 0),
                image_h=int(row["image_h"] or 0),
                outcome_keep=row["outcome_keep"] or "parse_error",
                outcome_drop=row["outcome_drop"] or "parse_error",
                iou_keep=float(row["iou_keep"] or 0.0),
                iou_drop=float(row["iou_drop"] or 0.0),
                n_elements_keep=int(row["n_elements_keep"] or 0),
                n_elements_drop=int(row["n_elements_drop"] or 0),
                prompt_tokens=int(row["prompt_tokens"] or 0),
                completion_tokens=int(row["completion_tokens"] or 0),
                latency_s=float(row["latency_s"] or 0.0),
                error=row["error"] or None,
            )
            rows.append(r)

    metrics = aggregate_metrics(rows, args.system_id, args.selector_model)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Screen2AX-Task | system={args.system_id} | selector={args.selector_model}")
    print(f"{'='*60}")
    print(f"n_samples:                   {metrics['n_samples']}")
    print(f"keep-groups success rate:    {metrics['keep_groups']['success_rate']:.4f}")
    print(f"  perception_error:          {metrics['keep_groups']['n_perception_error']}")
    print(f"  selection_error:           {metrics['keep_groups']['n_selection_error']}")
    print(f"  parse_error:               {metrics['keep_groups']['n_parse_error']}")
    print(f"drop-groups success rate:    {metrics['drop_groups']['success_rate']:.4f}")
    print(f"  perception_error:          {metrics['drop_groups']['n_perception_error']}")
    print(f"  selection_error:           {metrics['drop_groups']['n_selection_error']}")
    print(f"  parse_error:               {metrics['drop_groups']['n_parse_error']}")
    print(f"prompt tokens (total):       {metrics['total_prompt_tokens']:,}")
    print(f"completion tokens (total):   {metrics['total_completion_tokens']:,}")
    print(f"results: {csv_path}")
    print(f"metrics: {metrics_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Screen2AX-Task evaluation with GPT-5 selector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred-dir", required=True,
                   help="Directory with per-sample predictions: {id}.txt or {id}.json")
    p.add_argument("--pred-format", choices=["linearized_ax", "screen2ax_json"], required=True)
    p.add_argument("--pred-coords", choices=["normalized", "pixel"], default="normalized",
                   help="Are bbox coords normalized 0–1000 or already pixel coords?")
    p.add_argument("--gt-dir", default="/workspace/data/screen2ax_task/annotations")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--system-id", default="unnamed",
                   help="Label written into metrics.json")
    p.add_argument("--selector-model", default="gpt-5")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=None,
                   help="Smoke test cap; omit to evaluate all")
    p.add_argument("--resume", action="store_true",
                   help="Skip sample_ids already present in results.csv")
    p.add_argument("--save-prompts", action="store_true",
                   help="Save per-sample prompt+response snapshots to prompts/{id}.json")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    asyncio.run(run(args))
