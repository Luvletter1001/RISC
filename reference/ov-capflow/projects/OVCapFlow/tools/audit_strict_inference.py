import argparse
import ast
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from mmengine import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


FORBIDDEN = {
    'topk', 'nms', 'rotated_nms', 'multiclass_nms', 'minAreaRect'
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def scan_calls(root):
    hits = []
    for path in sorted(Path(root).rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) \
                else function.id if isinstance(function, ast.Name) else ''
            if name in FORBIDDEN:
                hits.append({
                    'path': str(path),
                    'line': node.lineno,
                    'call': name,
                })
    return hits


@contextmanager
def record_forbidden_runtime_calls():
    """Temporarily wrap selection/postprocess entry points and count calls."""
    patches = []
    counts = {}

    def patch(owner, attribute, label):
        if owner is None or not hasattr(owner, attribute):
            return
        original = getattr(owner, attribute)

        def wrapped(*args, **kwargs):
            counts[label] = counts.get(label, 0) + 1
            return original(*args, **kwargs)

        setattr(owner, attribute, wrapped)
        patches.append((owner, attribute, original))

    patch(torch, 'topk', 'torch.topk')
    patch(torch.Tensor, 'topk', 'Tensor.topk')
    try:
        import cv2
        patch(cv2, 'minAreaRect', 'cv2.minAreaRect')
    except ImportError:
        pass
    try:
        import mmcv.ops as mmcv_ops
        patch(mmcv_ops, 'nms', 'mmcv.ops.nms')
        patch(mmcv_ops, 'nms_rotated', 'mmcv.ops.nms_rotated')
    except ImportError:
        pass

    hits = []
    try:
        yield hits
    finally:
        for owner, attribute, original in reversed(patches):
            setattr(owner, attribute, original)
        hits.extend({
            'call': label,
            'count': count,
        } for label, count in sorted(counts.items()))


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    model = MODELS.build(cfg.model).to(args.device).eval()
    checkpoint_path = args.checkpoint or cfg.get('load_from')
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(
            checkpoint.get('state_dict', checkpoint), strict=False)
    dataloader = Runner.build_dataloader(cfg.val_dataloader)
    batch = next(iter(dataloader))
    batch = model.data_preprocessor(batch, training=False)
    with record_forbidden_runtime_calls() as runtime_hits:
        with torch.no_grad():
            predictions = model.predict(
                batch['inputs'], batch['data_samples'], rescale=True)
    counts = [len(sample.pred_instances) for sample in predictions]
    hits = scan_calls('projects/OVCapFlow/ov_capflow')
    report = {
        'pass': (not hits and not runtime_hits and
                 counts == [cfg.model.num_queries] * len(counts)),
        'num_matching_queries': int(cfg.model.num_queries),
        'predictions_per_image': counts,
        'forbidden_calls': hits,
        'runtime_forbidden_calls': runtime_hits,
        'uses_nms': any('nms' in hit['call'] for hit in hits),
        'uses_topk': any(hit['call'] == 'topk' for hit in hits),
        'uses_min_area_rect': any(
            hit['call'] == 'minAreaRect' for hit in hits),
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    if not report['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
