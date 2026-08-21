import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor


def summarize_image_mediators(
        ious: Tensor,
        null_logits: Optional[Tensor] = None,
        semantic_gate: Optional[Tensor] = None,
        iou_threshold: float = 0.5,
        scores: Optional[Tensor] = None) -> Dict[str, float]:
    """Return additive all-query mediator totals for one image."""
    if ious.ndim != 2:
        raise ValueError('ious must have shape [predictions, ground_truth]')
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError('iou_threshold must be in [0, 1]')

    num_predictions, num_gt = ious.shape
    if scores is not None and scores.shape != (num_predictions, ):
        raise ValueError('scores must have one value per prediction')
    matched = torch.zeros(
        num_predictions, dtype=torch.bool, device=ious.device)
    covered_gt = 0
    duplicate_extras = 0
    if num_gt:
        best_iou, best_gt = ious.max(dim=1)
        matched = best_iou >= iou_threshold
        assignments = torch.bincount(
            best_gt[matched], minlength=num_gt)
        covered_gt = int((assignments > 0).sum().item())
        duplicate_extras = int(
            (assignments - 1).clamp(min=0).sum().item())

    totals = {
        'image_count': 1,
        'prediction_count': int(num_predictions),
        'gt_count': int(num_gt),
        'covered_gt': covered_gt,
        'duplicate_extras': duplicate_extras,
        'matched_queries': int(matched.sum().item()),
        'unmatched_queries': int((~matched).sum().item()),
        'null_brier_sum': 0.0,
        'null_brier_count': 0,
        'matched_gate_sum': 0.0,
        'matched_gate_count': 0,
        'unmatched_gate_sum': 0.0,
        'unmatched_gate_count': 0,
        'empty_image_count': int(num_gt == 0),
        'empty_prediction_count': int(num_predictions if num_gt == 0 else 0),
        'empty_foreground_score_sum': 0.0,
        'empty_foreground_score_image_count': 0,
    }

    if num_gt == 0 and scores is not None:
        totals['empty_foreground_score_sum'] = float(scores.sum().item())
        totals['empty_foreground_score_image_count'] = 1

    if null_logits is not None:
        if null_logits.shape != (num_predictions, ):
            raise ValueError('null_logits must have one value per prediction')
        target_null = (~matched).to(null_logits.dtype)
        null_probability = null_logits.sigmoid()
        totals['null_brier_sum'] = float(
            ((null_probability - target_null)**2).sum().item())
        totals['null_brier_count'] = int(num_predictions)

    if semantic_gate is not None:
        if semantic_gate.shape[0] != num_predictions:
            raise ValueError(
                'semantic_gate must have one row per prediction')
        if semantic_gate.ndim == 1:
            gate_strength = semantic_gate.abs()
        elif semantic_gate.ndim == 2:
            gate_strength = semantic_gate.abs().mean(dim=-1)
        else:
            raise ValueError(
                'semantic_gate must be [predictions] or [predictions, dims]')
        totals['matched_gate_sum'] = float(
            gate_strength[matched].sum().item())
        totals['matched_gate_count'] = int(matched.sum().item())
        totals['unmatched_gate_sum'] = float(
            gate_strength[~matched].sum().item())
        totals['unmatched_gate_count'] = int((~matched).sum().item())
    return totals


def add_mediator_totals(target: Dict[str, float],
                        source: Dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def finalize_mediator_totals(totals: Dict[str, float]) -> Dict:
    """Convert additive totals into reportable dataset-level metrics."""
    gt_count = int(totals.get('gt_count', 0))
    null_count = int(totals.get('null_brier_count', 0))
    matched_gate_count = int(totals.get('matched_gate_count', 0))
    unmatched_gate_count = int(totals.get('unmatched_gate_count', 0))
    empty_image_count = int(totals.get('empty_image_count', 0))
    empty_score_image_count = int(
        totals.get('empty_foreground_score_image_count', 0))
    matched_gate_mean = (
        float(totals['matched_gate_sum']) / matched_gate_count
        if matched_gate_count else None)
    unmatched_gate_mean = (
        float(totals['unmatched_gate_sum']) / unmatched_gate_count
        if unmatched_gate_count else None)
    gate_gap = (
        unmatched_gate_mean - matched_gate_mean
        if matched_gate_mean is not None and
        unmatched_gate_mean is not None else None)
    return {
        'image_count': int(totals.get('image_count', 0)),
        'prediction_count': int(totals.get('prediction_count', 0)),
        'gt_count': gt_count,
        'matched_queries': int(totals.get('matched_queries', 0)),
        'unmatched_queries': int(totals.get('unmatched_queries', 0)),
        'gt_coverage': (
            float(totals.get('covered_gt', 0)) / gt_count
            if gt_count else None),
        'duplicate_extras_per_gt': (
            float(totals.get('duplicate_extras', 0)) / gt_count
            if gt_count else None),
        'null_brier': (
            float(totals.get('null_brier_sum', 0.0)) / null_count
            if null_count else None),
        'matched_gate_mean': matched_gate_mean,
        'unmatched_gate_mean': unmatched_gate_mean,
        'gate_gap': gate_gap,
        'empty_image_count': empty_image_count,
        'empty_predictions_per_image': (
            float(totals.get('empty_prediction_count', 0)) /
            empty_image_count if empty_image_count else None),
        'empty_foreground_score_mass_mean': (
            float(totals.get('empty_foreground_score_sum', 0.0)) /
            empty_score_image_count if empty_score_image_count else None),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    return parser.parse_args()


def _box_tensor(boxes) -> Tensor:
    return boxes.tensor if hasattr(boxes, 'tensor') else boxes


def main():
    from mmcv.ops import box_iou_rotated
    from mmengine import Config
    from mmengine.runner import Runner, load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmrotate.registry import MODELS
    from mmrotate.utils import register_all_modules

    args = parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model = model.cuda().eval()
    dataloader = Runner.build_dataloader(cfg.test_dataloader)

    totals: Dict[str, float] = {}
    prediction_counts = []
    capacity_mass_abs_error = 0.0
    capacity_mass_count = 0
    with torch.no_grad():
        for raw_batch in dataloader:
            batch = model.data_preprocessor(raw_batch, training=False)
            predictions = model.predict(
                batch['inputs'], batch['data_samples'], rescale=True)
            null_logits = model.decoder.last_null_logits
            last_layer = model.decoder.layers[-1]
            semantic_gate = last_layer.last_semantic_gate
            capacity = last_layer.last_capacity

            for index, (prediction, sample) in enumerate(zip(
                    predictions, batch['data_samples'])):
                pred_boxes = _box_tensor(prediction.pred_instances.bboxes)
                gt_boxes = _box_tensor(sample.gt_instances.bboxes)
                if gt_boxes.shape[0]:
                    ious = box_iou_rotated(pred_boxes, gt_boxes)
                else:
                    ious = pred_boxes.new_empty((pred_boxes.shape[0], 0))
                image_totals = summarize_image_mediators(
                    ious,
                    null_logits=(None if null_logits is None
                                 else null_logits[index]),
                    semantic_gate=(None if semantic_gate is None
                                   else semantic_gate[index]),
                    iou_threshold=args.iou_threshold,
                    scores=prediction.pred_instances.scores)
                add_mediator_totals(totals, image_totals)
                prediction_counts.append(int(pred_boxes.shape[0]))
                if capacity is not None:
                    capacity_mass_abs_error += abs(
                        float(capacity[index].sum().item()) -
                        int(gt_boxes.shape[0]))
                    capacity_mass_count += 1

    report = finalize_mediator_totals(totals)
    report.update({
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'iou_threshold': args.iou_threshold,
        'predictions_per_image_min': min(prediction_counts),
        'predictions_per_image_max': max(prediction_counts),
        'capacity_mass_mae': (
            capacity_mass_abs_error / capacity_mass_count
            if capacity_mass_count else None),
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
