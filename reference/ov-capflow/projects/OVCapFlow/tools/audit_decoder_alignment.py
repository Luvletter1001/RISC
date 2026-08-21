"""Audit decoder-row alignment with exported rotated predictions."""

import argparse
import json
from pathlib import Path

import torch
from mmengine import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def export_decoder_boxes(boxes, img_meta, angle_factor, rescale=True):
    """Apply OVCapFlow's documented coordinate transform without selection."""
    exported = boxes.clone()
    img_h, img_w = img_meta['img_shape'][:2]
    exported[:, 0:4:2] *= img_w
    exported[:, 1:4:2] *= img_h
    exported[:, 4] *= angle_factor
    exported[:, 0:4:2].clamp_(min=0, max=img_w)
    exported[:, 1:4:2].clamp_(min=0, max=img_h)
    if rescale:
        scale_factor = exported.new_tensor(img_meta['scale_factor'])
        if scale_factor.numel() == 2:
            scale_factor = scale_factor.repeat(2)
        if scale_factor.numel() != 4:
            raise ValueError('scale_factor must contain two or four values')
        scale_factor = torch.cat([scale_factor, scale_factor.new_ones(1)])
        exported /= scale_factor
    return exported


def align_decoder_predictions(decoder_boxes, prediction_boxes, img_meta,
                              angle_factor, expected_queries=600,
                              atol=1e-5, rtol=1e-5):
    expected = export_decoder_boxes(
        decoder_boxes, img_meta, angle_factor, rescale=True)
    same_shape = prediction_boxes.shape == expected.shape
    if same_shape and expected.numel():
        max_error = float((prediction_boxes - expected).abs().max().item())
    elif same_shape:
        max_error = 0.0
    else:
        max_error = None
    same_order = bool(same_shape and torch.allclose(
        prediction_boxes, expected, atol=atol, rtol=rtol))
    count = int(prediction_boxes.shape[0])
    return {
        'pass': bool(count == expected_queries and same_order),
        'expected_queries': int(expected_queries),
        'prediction_count': count,
        'decoder_count': int(decoder_boxes.shape[0]),
        'same_shape': bool(same_shape),
        'same_query_order': same_order,
        'max_abs_error': max_error,
    }


def _load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    model = MODELS.build(cfg.model).to(args.device).eval()
    checkpoint_path = args.checkpoint or cfg.get('load_from')
    missing_keys = []
    unexpected_keys = []
    if checkpoint_path:
        missing_keys, unexpected_keys = _load_checkpoint(
            model, checkpoint_path)

    captured = []

    def capture_head_output(_module, _inputs, output):
        captured.append(output[1][-1].detach())

    handle = model.bbox_head.register_forward_hook(capture_head_output)
    cfg.val_dataloader.num_workers = 0
    dataloader = Runner.build_dataloader(cfg.val_dataloader)
    batch = next(iter(dataloader))
    batch = model.data_preprocessor(batch, training=False)
    try:
        with torch.no_grad():
            predictions = model.predict(
                batch['inputs'], batch['data_samples'], rescale=True)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f'expected one bbox-head forward, observed {len(captured)}')

    layer_boxes = captured[0]
    image_reports = []
    for decoder_boxes, prediction, sample in zip(
            layer_boxes, predictions, batch['data_samples']):
        predicted_boxes = prediction.pred_instances.bboxes
        if hasattr(predicted_boxes, 'tensor'):
            predicted_boxes = predicted_boxes.tensor
        image_reports.append(align_decoder_predictions(
            decoder_boxes,
            predicted_boxes,
            sample.metainfo,
            angle_factor=model.bbox_head.angle_factor,
            expected_queries=int(cfg.model.num_queries)))
    report = {
        'pass': bool(image_reports and all(
            item['pass'] for item in image_reports)),
        'config': str(args.config),
        'checkpoint': str(checkpoint_path) if checkpoint_path else None,
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'images': image_reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
