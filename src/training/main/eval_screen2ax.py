"""
F1@IoU evaluation for Screen2AX element detection.

Computes element-level F1 score by matching predicted and ground-truth
bounding boxes using IoU threshold (default 0.1).
"""
import re
import os
import torch
import wandb
import tqdm


def parse_elements(text):
    """Parse 'AXButton [x1, y1, x2, y2]; ...' into list of (type, [x1,y1,x2,y2])."""
    pattern = r'(\w+)\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    elements = []
    for role, x1, y1, x2, y2 in re.findall(pattern, text):
        elements.append((role, [int(x1), int(y1), int(x2), int(y2)]))
    return elements


def compute_iou(box_a, box_b):
    """IoU between two [x1, y1, x2, y2] boxes (0-1000 scale)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_elements(pred_elements, gt_elements, iou_threshold=0.1):
    """
    Greedy matching of predicted to GT elements.
    A match requires same class AND IoU >= threshold.
    Returns (tp, fp, fn).
    """
    matched_gt = set()
    tp = 0

    for p_type, p_box in pred_elements:
        best_iou = 0
        best_gt_idx = -1
        for g_idx, (g_type, g_box) in enumerate(gt_elements):
            if g_idx in matched_gt:
                continue
            if p_type != g_type:
                continue
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_elements) - tp
    fn = len(gt_elements) - tp
    return tp, fp, fn


def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def validate_screen2ax(val_loader, model_engine, processor, epoch, global_step, writer, args, split="val"):
    """Validation with F1@IoU=0.1 using model.generate(), distributed across all GPUs."""
    import torch.distributed as dist
    model_engine.eval()

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    total_tp, total_fp, total_fn = 0, 0, 0
    total_loss = 0.0
    n_samples = 0

    for input_dict in tqdm.tqdm(val_loader, desc=split.capitalize(), disable=(args.global_rank != 0)):
        torch.cuda.empty_cache()
        from data.data_utils import dict_to_cuda
        input_dict = dict_to_cuda(input_dict, device=f'cuda:{local_rank}')

        if args.precision == "fp16":
            input_dict["pixel_values"] = input_dict["pixel_values"].half()
        elif args.precision == "bf16":
            input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
        else:
            input_dict["pixel_values"] = input_dict["pixel_values"].float()

        forward_dict = dict(
            pixel_values=input_dict["pixel_values"],
            input_ids=input_dict["input_ids"],
            labels=input_dict["labels"],
        )
        if "image_sizes" in input_dict:
            forward_dict["image_grid_thw"] = input_dict["image_sizes"]
        for key in ["patch_assign", "patch_assign_len", "patch_pos", "select_mask"]:
            if key in input_dict:
                forward_dict[key] = input_dict[key]

        with torch.no_grad():
            # Compute val loss
            output_dict = model_engine(**forward_dict, output_hidden_states=True)
            total_loss += output_dict['loss'].item()

            # Generate predictions
            try:
                generate_ids = model_engine.generate(
                    **forward_dict,
                    max_new_tokens=args.max_new_tokens if hasattr(args, 'max_new_tokens') else 2048,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
                generate_ids = generate_ids[:, input_dict['input_ids'].shape[1]:]
                pred_text = processor.batch_decode(
                    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0]
            except Exception as e:
                print(f"Generate error: {e}")
                pred_text = ""

        meta = input_dict['meta_data'][0]
        gt_text = meta.get('all_elements_str', '')

        pred_elements = parse_elements(pred_text)
        gt_elements = parse_elements(gt_text)

        tp, fp, fn = match_elements(pred_elements, gt_elements, iou_threshold=0.1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        n_samples += 1

    # Aggregate tp/fp/fn/loss/n_samples across all ranks
    if args.distributed:
        stats = torch.tensor(
            [total_tp, total_fp, total_fn, total_loss, n_samples],
            dtype=torch.float64, device=f'cuda:{local_rank}'
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_tp   = int(stats[0].item())
        total_fp   = int(stats[1].item())
        total_fn   = int(stats[2].item())
        total_loss = stats[3].item()
        n_samples  = int(stats[4].item())

    precision, recall, f1 = compute_f1(total_tp, total_fp, total_fn)
    avg_loss = total_loss / max(n_samples, 1)

    if args.global_rank == 0:
        print(f"\n[{split.capitalize()}] Epoch {epoch} | Loss: {avg_loss:.4f} | "
              f"F1@IoU=0.1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | "
              f"TP: {total_tp} FP: {total_fp} FN: {total_fn}")

        if not args.debug:
            writer.add_scalar(f"{split}/loss", avg_loss, global_step)
            writer.add_scalar(f"{split}/f1_iou01", f1, global_step)
            writer.add_scalar(f"{split}/precision", precision, global_step)
            writer.add_scalar(f"{split}/recall", recall, global_step)
            wandb.log({
                f"{split}/loss": avg_loss,
                f"{split}/f1_iou01": f1,
                f"{split}/precision": precision,
                f"{split}/recall": recall,
                f"{split}/tp": total_tp,
                f"{split}/fp": total_fp,
                f"{split}/fn": total_fn,
            }, step=global_step)

    # Return score for best-model checkpointing (higher = better)
    return f1


def validate_screen2ax_tree(val_loader, model_engine, processor, epoch, global_step, writer, args, split="val"):
    """Validation for linearized AX-tree task. Reports val loss only."""
    import torch.distributed as dist
    model_engine.eval()

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    total_loss = 0.0
    n_samples = 0

    for input_dict in tqdm.tqdm(val_loader, desc=split.capitalize(), disable=(args.global_rank != 0)):
        torch.cuda.empty_cache()
        from data.data_utils import dict_to_cuda
        input_dict = dict_to_cuda(input_dict, device=f'cuda:{local_rank}')

        if args.precision == "fp16":
            input_dict["pixel_values"] = input_dict["pixel_values"].half()
        elif args.precision == "bf16":
            input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
        else:
            input_dict["pixel_values"] = input_dict["pixel_values"].float()

        forward_dict = dict(
            pixel_values=input_dict["pixel_values"],
            input_ids=input_dict["input_ids"],
            labels=input_dict["labels"],
        )
        if "image_sizes" in input_dict:
            forward_dict["image_grid_thw"] = input_dict["image_sizes"]
        for key in ["patch_assign", "patch_assign_len", "patch_pos", "select_mask"]:
            if key in input_dict:
                forward_dict[key] = input_dict[key]

        with torch.no_grad():
            output_dict = model_engine(**forward_dict)
            total_loss += output_dict['loss'].item()
        n_samples += 1

    if args.distributed:
        stats = torch.tensor(
            [total_loss, n_samples],
            dtype=torch.float64, device=f'cuda:{local_rank}'
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = stats[0].item()
        n_samples  = int(stats[1].item())

    avg_loss = total_loss / max(n_samples, 1)

    if args.global_rank == 0:
        print(f"\n[{split.capitalize()}] Epoch {epoch} | Loss: {avg_loss:.4f}")
        if not args.debug:
            writer.add_scalar(f"{split}/loss", avg_loss, global_step)
            wandb.log({f"{split}/loss": avg_loss}, step=global_step)

    # Lower loss = better; return negative so checkpointing logic (higher=better) works
    return -avg_loss
