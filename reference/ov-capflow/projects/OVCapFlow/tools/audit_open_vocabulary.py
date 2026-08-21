"""Audit prompt-shape independence for OVCapFlow."""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from mmengine import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


FORBIDDEN_CHECKPOINT_TOKENS = (
    'teacher', 'distill', 'pseudo', 'dense_head', 'rpn', 'roi_head')
CLASSIFIER_NAME_TOKENS = (
    'classifier', 'classification', 'class_embed', 'cls_score')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def state_shapes(model):
    return {
        name: list(value.shape)
        for name, value in model.state_dict().items()
    }


def find_fixed_classification_layers(model, class_count):
    """Find trainable named classifier linears fixed to class count."""
    hits = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.out_features != class_count:
            continue
        if not any(parameter.requires_grad for parameter in module.parameters()):
            continue
        lower_name = name.lower()
        if any(token in lower_name for token in CLASSIFIER_NAME_TOKENS):
            hits.append(name)
    return sorted(hits)


def find_forbidden_checkpoint_keys(keys):
    return [
        key for key in keys
        if any(token in key.lower() for token in FORBIDDEN_CHECKPOINT_TOKENS)
    ]


def _character_spans(prompt, entities):
    spans = []
    cursor = 0
    for entity in entities:
        start = prompt.find(entity, cursor)
        if start < 0:
            raise ValueError(f'entity {entity!r} is missing from prompt')
        end = start + len(entity)
        spans.append([[start, end]])
        cursor = end
    return spans


def _serializable_positive_map(positive_map):
    serialized = {}
    for label, indices in positive_map.items():
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().tolist()
        serialized[str(label)] = [int(index) for index in indices]
    return serialized


def set_prompt_metainfo(sample, entities):
    """Replace an inference sample's prompt through MMEngine metainfo."""
    sample.set_metainfo({
        'text': list(entities),
        'custom_entities': True,
    })


def audit_prompt_variants(model, variants, run_variant=None,
                          expected_queries=600, class_count=18,
                          checkpoint_keys=()):
    """Audit actual token protocol and optional inference for prompt variants."""
    before = state_shapes(model)
    records = {}
    count_failures = []
    for variant_name, requested_entities in variants.items():
        (positive_map, actual_prompt, _dense_positive_map,
         actual_entities) = model.get_tokens_positive_and_prompts(
             list(requested_entities), custom_entities=True)
        entity_order = list(actual_entities)
        result_counts = [] if run_variant is None else [
            int(count) for count in run_variant(list(requested_entities))]
        if run_variant is not None and any(
                count != expected_queries for count in result_counts):
            count_failures.append(variant_name)
        records[variant_name] = {
            'requested_entities': list(requested_entities),
            'entity_order': entity_order,
            'actual_prompt': actual_prompt,
            'token_spans': _character_spans(actual_prompt, entity_order),
            'positive_map': _serializable_positive_map(positive_map),
            'result_counts': result_counts,
        }
    after = state_shapes(model)
    fixed_classifiers = find_fixed_classification_layers(model, class_count)
    forbidden_keys = find_forbidden_checkpoint_keys(checkpoint_keys)
    shape_unchanged = before == after
    return {
        'pass': bool(shape_unchanged and not count_failures and
                     not fixed_classifiers and not forbidden_keys),
        'expected_queries': int(expected_queries),
        'state_shapes_unchanged': shape_unchanged,
        'state_shapes_before': before,
        'state_shapes_after': after,
        'fixed_classification_layers': fixed_classifiers,
        'forbidden_checkpoint_keys': forbidden_keys,
        'count_failures': count_failures,
        'variants': records,
    }


def _load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    return (list(state_dict), list(incompatible.missing_keys),
            list(incompatible.unexpected_keys))


def _canonical_classes(cfg):
    if 'classes' in cfg:
        return list(cfg.classes)
    return list(cfg.val_dataloader.dataset.metainfo.classes)


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    model = MODELS.build(cfg.model).to(args.device).eval()
    checkpoint_path = args.checkpoint or cfg.get('load_from')
    checkpoint_keys = []
    missing_keys = []
    unexpected_keys = []
    if checkpoint_path:
        checkpoint_keys, missing_keys, unexpected_keys = _load_checkpoint(
            model, checkpoint_path)

    dataloader = Runner.build_dataloader(cfg.val_dataloader)
    batch = next(iter(dataloader))
    batch = model.data_preprocessor(batch, training=False)

    def run_variant(entities):
        samples = copy.deepcopy(batch['data_samples'])
        for sample in samples:
            set_prompt_metainfo(sample, entities)
        with torch.no_grad():
            predictions = model.predict(
                batch['inputs'], samples, rescale=True)
        return [len(sample.pred_instances) for sample in predictions]

    canonical = _canonical_classes(cfg)
    synonyms = {
        'airport': 'airport airfield',
        'plane': 'plane aircraft',
        'ship': 'ship vessel',
    }
    variants = {
        'canonical': canonical,
        'reordered': list(reversed(canonical)),
        'synonym_expanded': [synonyms.get(name, name) for name in canonical],
        'one_new_class': canonical + ['wind-turbine'],
    }
    report = audit_prompt_variants(
        model,
        variants,
        run_variant=run_variant,
        expected_queries=int(cfg.model.num_queries),
        class_count=len(canonical),
        checkpoint_keys=checkpoint_keys)
    report.update(
        config=str(args.config),
        checkpoint=str(checkpoint_path) if checkpoint_path else None,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
