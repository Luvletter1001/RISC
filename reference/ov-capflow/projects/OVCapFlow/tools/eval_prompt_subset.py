"""Evaluate a checkpoint with all, base-only, or novel-only prompts."""

import argparse
import copy
import json
from pathlib import Path

import torch
from mmengine import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import METRICS, MODELS
from mmrotate.utils import register_all_modules


def select_prompt_classes(all_classes, novel_classes, mode):
    all_classes = tuple(all_classes)
    novel_classes = tuple(novel_classes)
    if mode == 'all':
        return all_classes
    if mode == 'novel':
        return novel_classes
    if mode == 'base':
        novel_set = set(novel_classes)
        return tuple(name for name in all_classes if name not in novel_set)
    raise ValueError(f'unsupported prompt mode: {mode}')


def configure_prompt_subset(cfg, classes):
    """Prepare deterministic test loading while retaining the 18-class parser."""
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    cfg.test_dataloader.dataset.filter_cfg.filter_empty_gt = False
    cfg.test_dataloader.dataset.test_mode = True


def remap_sample_ground_truth(sample, all_classes, subset_classes):
    """Filter full-vocabulary GT and remap labels into prompt-subset order."""
    remapped = copy.deepcopy(sample)
    subset_index = {
        name: index for index, name in enumerate(subset_classes)
    }
    label_mapping = torch.full(
        (len(all_classes), ), -1, dtype=torch.long)
    for original_index, name in enumerate(all_classes):
        if name in subset_index:
            label_mapping[original_index] = subset_index[name]

    for field in ('gt_instances', 'ignored_instances'):
        instances = getattr(remapped, field)
        labels = instances.labels
        mapping = label_mapping.to(labels.device)
        new_labels = mapping[labels]
        keep = new_labels >= 0
        instances = instances[keep]
        instances.labels = new_labels[keep]
        setattr(remapped, field, instances)
    return remapped


def metric_sample_dicts(samples):
    return [sample.to_dict() for sample in samples]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--mode', choices=('all', 'base', 'novel'),
                        required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, 'item'):
        return value.item()
    return value


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    classes = select_prompt_classes(
        cfg.classes, cfg.novel_classes, args.mode)
    configure_prompt_subset(cfg, classes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model = model.cuda().eval()
    dataloader = Runner.build_dataloader(cfg.test_dataloader)
    metric = METRICS.build(cfg.test_evaluator)
    metric.dataset_meta = dict(classes=tuple(classes))
    all_classes = tuple(cfg.classes)
    prediction_counts = []
    with torch.no_grad():
        for raw_batch in dataloader:
            batch = model.data_preprocessor(raw_batch, training=False)
            for sample in batch['data_samples']:
                sample.set_metainfo({
                    'text': list(classes),
                    'custom_entities': True,
                })
            predictions = model.predict(
                batch['inputs'], batch['data_samples'], rescale=True)
            remapped = [
                remap_sample_ground_truth(sample, all_classes, classes)
                for sample in predictions
            ]
            prediction_counts.extend(
                len(sample.pred_instances) for sample in remapped)
            metric.process(raw_batch, metric_sample_dicts(remapped))
    metrics = metric.compute_metrics(metric.results)
    report = {
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'mode': args.mode,
        'classes': list(classes),
        'class_count': len(classes),
        'metrics': _json_value(metrics),
        'image_count': len(metric.results),
        'predictions_per_image_min': min(prediction_counts),
        'predictions_per_image_max': max(prediction_counts),
    }
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
