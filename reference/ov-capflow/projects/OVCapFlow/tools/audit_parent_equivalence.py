import argparse
import copy
import json
from pathlib import Path

import torch
from mmengine import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('parent_config')
    parser.add_argument('candidate_config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def tensor_report(parent, candidate):
    report = {
        'parent_shape': list(parent.shape),
        'candidate_shape': list(candidate.shape),
        'exact': False,
        'max_abs_diff': None,
    }
    if parent.shape != candidate.shape:
        return report
    report['exact'] = bool(torch.equal(parent, candidate))
    if report['exact']:
        report['max_abs_diff'] = 0.0
        return report

    jointly_finite = torch.isfinite(parent) & torch.isfinite(candidate)
    nonfinite = ~jointly_finite
    if nonfinite.any() and not torch.equal(
            parent[nonfinite], candidate[nonfinite]):
        return report
    if jointly_finite.any():
        report['max_abs_diff'] = float(
            (parent[jointly_finite].float() -
             candidate[jointly_finite].float()).abs().max().item())
    else:
        report['max_abs_diff'] = 0.0
    return report


def _capture(config_path, checkpoint_path, raw_batch):
    cfg = Config.fromfile(config_path)
    import_modules_from_strings(**cfg.custom_imports)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu', strict=False)
    model = model.cuda().eval()
    captured = {}

    def capture_decoder(module, inputs, output):
        captured['decoder_states'] = output[0].detach().cpu()
        references = output[1]
        if isinstance(references, (list, tuple)):
            references = torch.stack(references)
        captured['references'] = references.detach().cpu()

    def capture_head(module, inputs, output):
        captured['classification_scores'] = output[0].detach().cpu()
        captured['rotated_boxes'] = output[1].detach().cpu()

    handles = [
        model.decoder.register_forward_hook(capture_decoder),
        model.bbox_head.register_forward_hook(capture_head),
    ]
    batch = model.data_preprocessor(copy.deepcopy(raw_batch), training=False)
    with torch.no_grad():
        predictions = model.predict(
            batch['inputs'], batch['data_samples'], rescale=True)
    for handle in handles:
        handle.remove()

    captured['prediction_counts'] = [
        len(sample.pred_instances) for sample in predictions
    ]
    captured['prediction_scores'] = torch.cat([
        sample.pred_instances.scores.detach().cpu() for sample in predictions
    ])
    captured['prediction_labels'] = torch.cat([
        sample.pred_instances.labels.detach().cpu() for sample in predictions
    ])
    captured['prediction_boxes'] = torch.cat([
        (sample.pred_instances.bboxes.tensor if hasattr(
            sample.pred_instances.bboxes, 'tensor') else
         sample.pred_instances.bboxes).detach().cpu() for sample in predictions
    ])
    del model
    torch.cuda.empty_cache()
    return captured


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    parent_cfg = Config.fromfile(args.parent_config)
    import_modules_from_strings(**parent_cfg.custom_imports)
    parent_cfg.val_dataloader.num_workers = 0
    parent_cfg.val_dataloader.persistent_workers = False
    dataloader = Runner.build_dataloader(parent_cfg.val_dataloader)
    raw_batch = next(iter(dataloader))

    parent = _capture(args.parent_config, args.checkpoint, raw_batch)
    candidate = _capture(args.candidate_config, args.checkpoint, raw_batch)
    tensor_names = [
        'decoder_states',
        'references',
        'classification_scores',
        'rotated_boxes',
        'prediction_scores',
        'prediction_labels',
        'prediction_boxes',
    ]
    comparisons = {
        name: tensor_report(parent[name], candidate[name])
        for name in tensor_names
    }
    parent_counts = parent['prediction_counts']
    candidate_counts = candidate['prediction_counts']
    passed = (
        all(item['exact'] for item in comparisons.values())
        and parent_counts == candidate_counts
        and all(count == parent_cfg.model.num_queries
                for count in parent_counts))
    report = {
        'pass': passed,
        'parent_config': str(Path(args.parent_config).resolve()),
        'candidate_config': str(Path(args.candidate_config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'parent_prediction_counts': parent_counts,
        'candidate_prediction_counts': candidate_counts,
        'comparisons': comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
