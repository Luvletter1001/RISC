"""Prepare an auditable generic GroundingDINO initialization checkpoint.

The output contains only tensors whose converted key and shape exactly match
the target OV-CapFlow model. Training state and remote-sensing-derived model
namespaces are rejected before anything is written.
"""

import argparse
import copy
import hashlib
import json
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

import torch
from mmengine import Config
from mmengine.utils import import_modules_from_strings
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


FORBIDDEN_SOURCE_PARTS = (
    'dota', 'openrsd', 'hrsc', 'fair1m', 'dior', 'p126', 'p121',
    'teacher', 'distill', 'pseudo')
FORBIDDEN_NAMESPACE_PARTS = (
    'teacher', 'distill', 'pseudo', 'dense_head', 'rpn_head', 'roi_head')
FORBIDDEN_TRAINING_KEYS = (
    'optimizer', 'optim_wrapper', 'param_schedulers', 'scheduler',
    'ema_state_dict', 'ema')
REQUIRED_PREFIXES = (
    'backbone.', 'encoder.', 'decoder.', 'language_model.', 'text_feat_map.')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reject_forbidden_source(path: Path) -> None:
    normalized = str(Path(path).expanduser().resolve()).lower()
    hits = sorted(part for part in FORBIDDEN_SOURCE_PARTS
                  if part in normalized)
    if hits:
        raise ValueError(
            'forbidden checkpoint source: ' + ', '.join(hits))


def reject_forbidden_namespaces(state_dict: Mapping[str, Tensor]) -> None:
    hits = sorted(
        key for key in state_dict
        if any(part in key.lower() for part in FORBIDDEN_NAMESPACE_PARTS))
    if hits:
        raise ValueError(
            'forbidden checkpoint namespace: ' + ', '.join(hits))


def unwrap_model_state(checkpoint: Mapping) -> Dict[str, Tensor]:
    training_hits = sorted(
        key for key in checkpoint
        if key.lower() in FORBIDDEN_TRAINING_KEYS)
    if training_hits:
        raise ValueError(
            'checkpoint contains forbidden training state: ' +
            ', '.join(training_hits))

    if 'model' in checkpoint:
        state = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint
    if not isinstance(state, Mapping):
        raise TypeError('checkpoint model state must be a mapping')

    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise TypeError('model state must map string keys to tensors')
        normalized[key[7:] if key.startswith('module.') else key] = value
    return normalized


def _correct_unfold_reduction_order(value: Tensor) -> Tensor:
    out_channel, in_channel = value.shape
    value = value.reshape(out_channel, 4, in_channel // 4)
    return value[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(
        out_channel, in_channel)


def _correct_unfold_norm_order(value: Tensor) -> Tensor:
    in_channel = value.shape[0]
    value = value.reshape(4, in_channel // 4)
    return value[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)


def _convert_key_value(key: str, value: Tensor) -> Tuple[str, Tensor]:
    """Apply the official OpenMMLab GroundingDINO conversion rules."""
    original_key = key
    if not key.startswith('module.'):
        key = 'module.' + key
    if 'module.bbox_embed' in key:
        key = key.replace(
            'module.bbox_embed', 'module.transformer.decoder.bbox_embed')

    if 'module.backbone.0' in key:
        converted = key.replace('module.backbone.0', 'backbone')
        if 'patch_embed.proj' in converted:
            converted = converted.replace(
                'patch_embed.proj', 'patch_embed.projection')
        elif 'pos_drop' in converted:
            converted = converted.replace('pos_drop', 'drop_after_pos')
        if 'layers' in converted:
            converted = converted.replace('layers', 'stages')
            if 'mlp.fc1' in converted:
                converted = converted.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in converted:
                converted = converted.replace('mlp.fc2', 'ffn.layers.1')
            elif 'attn' in converted:
                converted = converted.replace('attn', 'attn.w_msa')
            if 'downsample' in key:
                if 'reduction.' in key:
                    value = _correct_unfold_reduction_order(value)
                elif 'norm.' in key:
                    value = _correct_unfold_norm_order(value)
        return converted, value

    if 'module.bert' in key:
        return key.replace(
            'module.bert',
            'language_model.language_backbone.body.model'), value
    if 'module.feat_map' in key:
        return key.replace('module.feat_map', 'text_feat_map'), value
    if 'module.input_proj' in key:
        converted = key.replace('module.input_proj', 'neck.convs')
        converted = converted.replace(
            'neck.convs.3', 'neck.extra_convs.0')
        for source, target in (
                ('0.weight', 'conv.weight'), ('0.bias', 'conv.bias'),
                ('1.weight', 'gn.weight'), ('1.bias', 'gn.bias')):
            if source in converted:
                converted = converted.replace(source, target)
        return converted, value
    if 'module.transformer.level_embed' in key:
        return key.replace(
            'module.transformer.level_embed', 'level_embed'), value
    if 'module.transformer.encoder' in key:
        converted = key.replace('module.transformer.encoder', 'encoder')
        for source, target in (
                ('norm1', 'norms.0'), ('norm2', 'norms.1'),
                ('norm3', 'norms.2'),
                ('linear1', 'ffn.layers.0.0'),
                ('linear2', 'ffn.layers.1')):
            if source in converted:
                converted = converted.replace(source, target)
        if 'text_layers' in converted and 'self_attn' in converted:
            converted = converted.replace('self_attn', 'self_attn.attn')
        return converted, value
    if 'module.transformer.enc_output' in key:
        if 'module.transformer.enc_output_norm' in key:
            return key.replace(
                'module.transformer.enc_output_norm',
                'memory_trans_norm'), value
        return key.replace(
            'module.transformer.enc_output', 'memory_trans_fc'), value
    if 'module.transformer.enc_out_bbox_embed.layers' in key:
        for layer_id, linear_id in (('0', '0'), ('1', '2'), ('2', '4')):
            source = 'module.transformer.enc_out_bbox_embed.layers.' + layer_id
            if source in key:
                return key.replace(
                    source,
                    'bbox_head.reg_branches.6.' + linear_id), value
    if 'module.transformer.tgt_embed' in key:
        return key.replace(
            'module.transformer.tgt_embed', 'query_embedding'), value
    if 'module.transformer.decoder' in key:
        converted = key.replace('module.transformer.decoder', 'decoder')
        for source, target in (
                ('norm1', 'norms.2'), ('catext_norm', 'norms.1'),
                ('norm2', 'norms.0'), ('norm3', 'norms.3')):
            if source in converted:
                converted = converted.replace(source, target)
        if 'ca_text' in converted:
            converted = converted.replace('ca_text', 'cross_attn_text')
            for source, target in (
                    ('in_proj_weight', 'attn.in_proj_weight'),
                    ('in_proj_bias', 'attn.in_proj_bias'),
                    ('out_proj.weight', 'attn.out_proj.weight'),
                    ('out_proj.bias', 'attn.out_proj.bias')):
                if source in converted:
                    converted = converted.replace(source, target)
        if 'linear1' in converted:
            converted = converted.replace('linear1', 'ffn.layers.0.0')
        if 'linear2' in converted:
            converted = converted.replace('linear2', 'ffn.layers.1')
        if 'self_attn' in converted:
            converted = converted.replace('self_attn', 'self_attn.attn')
        if 'bbox_embed' in converted:
            parts = converted.split('.')
            reg_layer_id = int(parts[2])
            linear_id = int(parts[4])
            converted = (
                'bbox_head.reg_branches.' + str(reg_layer_id) + '.' +
                str(2 * linear_id) + '.' + parts[-1])
        return converted, value

    raise KeyError('unsupported GroundingDINO key: ' + original_key)


def convert_groundingdino_state_dict(
        state_dict: Mapping[str, Tensor]) -> OrderedDict:
    converted: MutableMapping[str, Tensor] = {}
    for key, value in state_dict.items():
        try:
            new_key, new_value = _convert_key_value(key, value)
        except KeyError:
            continue
        if new_key in converted:
            if torch.equal(converted[new_key], new_value):
                continue
            raise ValueError(
                'conflicting converted checkpoint key: ' + new_key)
        converted[new_key] = new_value
    return OrderedDict((key, converted[key]) for key in sorted(converted))


def filter_compatible_state_dict(
        source: Mapping[str, Tensor],
        target: Mapping[str, Tensor]) -> Tuple[OrderedDict, dict]:
    compatible = OrderedDict()
    shape_mismatches = []
    unexpected_keys = []
    for key in sorted(source):
        if key not in target:
            unexpected_keys.append(key)
            continue
        source_shape = list(source[key].shape)
        target_shape = list(target[key].shape)
        if source_shape != target_shape:
            shape_mismatches.append({
                'key': key,
                'source_shape': source_shape,
                'target_shape': target_shape,
            })
            continue
        compatible[key] = source[key]

    missing_keys = sorted(set(target) - set(compatible))
    matched_numel = sum(value.numel() for value in compatible.values())
    target_numel = sum(value.numel() for value in target.values())
    report = {
        'source_key_count': len(source),
        'target_key_count': len(target),
        'matched_key_count': len(compatible),
        'matched_numel': matched_numel,
        'target_numel': target_numel,
        'coverage_ratio': (
            matched_numel / target_numel if target_numel else 0.0),
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'shape_mismatches': shape_mismatches,
    }
    return compatible, report


def transport_first_content_queries(
        source: Mapping[str, Tensor],
        compatible: Mapping[str, Tensor],
        target: Mapping[str, Tensor],
        count: int = 600) -> Tuple[OrderedDict, dict]:
    """Transport the first generic content-query rows into fixed Q queries.

    This is deliberately narrower than general checkpoint conversion: the
    source and target keys are fixed, rows must be a contiguous prefix, and an
    already compatible target key is rejected rather than overwritten.
    """
    source_key = 'query_embedding.weight'
    target_key = 'query_initializer.query_embedding.weight'
    if source_key not in source:
        raise KeyError('source query key is missing: ' + source_key)
    if target_key not in target:
        raise KeyError('target query key is missing: ' + target_key)
    if target_key in compatible:
        raise ValueError('target query key is already present: ' + target_key)
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError('query transport count must be a positive integer')

    source_value = source[source_key]
    target_value = target[target_key]
    source_shape = list(source_value.shape)
    target_shape = list(target_value.shape)
    if (source_value.ndim != 2 or source_value.shape[0] < count or
            target_value.ndim != 2 or
            source_value.shape[1] != target_value.shape[1]):
        raise ValueError(
            'source query shape cannot supply target: source=' +
            str(source_shape) + ', target=' + str(target_shape) +
            ', count=' + str(count))
    expected_target_shape = [count, source_value.shape[1]]
    if target_shape != expected_target_shape:
        raise ValueError(
            'target query shape must equal transported shape: expected=' +
            str(expected_target_shape) + ', got=' + str(target_shape))
    if source_value.dtype != target_value.dtype:
        raise ValueError(
            'source and target query dtypes differ: source=' +
            str(source_value.dtype) + ', target=' + str(target_value.dtype))

    transported = dict(compatible)
    transported[target_key] = source_value[:count].clone()
    transported = OrderedDict(
        (key, transported[key]) for key in sorted(transported))
    report = {
        'enabled': True,
        'source_key': source_key,
        'target_key': target_key,
        'source_shape': source_shape,
        'transported_shape': expected_target_shape,
        'selection': 'first_contiguous_rows',
        'start_row_inclusive': 0,
        'end_row_exclusive': count,
    }
    return transported, report


def require_prefix_coverage(
        state_dict: Mapping[str, Tensor],
        prefixes: Sequence[str] = REQUIRED_PREFIXES) -> None:
    missing = [
        prefix for prefix in prefixes
        if not any(key.startswith(prefix) for key in state_dict)
    ]
    if missing:
        raise ValueError(
            'compatible checkpoint lacks required prefixes: ' +
            ', '.join(missing))


def write_provenance(path: Path, report: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')


def _git_commit() -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip()


def _target_state_from_config(config_path: Path) -> Mapping[str, Tensor]:
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config_path))
    if 'custom_imports' in cfg:
        import_modules_from_strings(**cfg.custom_imports)
    model_cfg = copy.deepcopy(cfg.model)
    if 'backbone' in model_cfg:
        model_cfg['backbone']['init_cfg'] = None
    model = MODELS.build(model_cfg)
    return model.state_dict()


def prepare_checkpoint(source: Path, config: Path, output: Path,
                       provenance: Path, expected_sha256: str,
                       transport_first_queries: bool = False) -> dict:
    source = Path(source).expanduser().resolve()
    config = Path(config).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    provenance = Path(provenance).expanduser().resolve()
    reject_forbidden_source(source)
    actual_source_sha = sha256_file(source)
    if actual_source_sha != expected_sha256.lower():
        raise ValueError(
            'source SHA256 mismatch: expected ' + expected_sha256.lower() +
            ', got ' + actual_source_sha)

    checkpoint = torch.load(str(source), map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise TypeError('checkpoint root must be a mapping')
    original_state = unwrap_model_state(checkpoint)
    reject_forbidden_namespaces(original_state)
    converted_state = convert_groundingdino_state_dict(original_state)
    reject_forbidden_namespaces(converted_state)

    target_state = _target_state_from_config(config)
    compatible, compatibility = filter_compatible_state_dict(
        converted_state, target_state)
    query_transport = {'enabled': False}
    if transport_first_queries:
        pre_transport_compatibility = compatibility
        compatible, query_transport = transport_first_content_queries(
            converted_state, compatible, target_state, count=600)
        _, compatibility = filter_compatible_state_dict(
            compatible, target_state)
        compatibility['pre_transport'] = pre_transport_compatibility
    require_prefix_coverage(compatible)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': compatible}, str(output))
    report = {
        'schema_version': 1,
        'git_commit': _git_commit(),
        'source_path': str(source),
        'source_sha256': actual_source_sha,
        'config_path': str(config),
        'config_sha256': sha256_file(config),
        'output_path': str(output),
        'output_sha256': sha256_file(output),
        'checkpoint_payload_keys': ['state_dict'],
        'optimizer_state_present': False,
        'scheduler_state_present': False,
        'ema_state_present': False,
        'forbidden_hits': [],
        'required_prefixes': list(REQUIRED_PREFIXES),
        'query_transport': query_transport,
        'fresh_parameter_rules': [
            (
                'Q600 content queries are the exact first 600 contiguous '
                'rows of generic Q900 tgt_embed'
                if transport_first_queries else
                'Q600 fixed query initializer is not copied from Q900 source'
            ),
            'incompatible horizontal 4-D regression tensors are omitted',
            'unmatched rotated-angle and geometry tensors use seeded init',
        ],
        **compatibility,
    }
    write_provenance(provenance, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('provenance', type=Path)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument(
        '--transport-first-queries',
        action='store_true',
        help='copy the exact first 600 generic content-query rows into Q600')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_checkpoint(
        source=args.source,
        config=args.config,
        output=args.output,
        provenance=args.provenance,
        expected_sha256=args.expected_sha256,
        transport_first_queries=args.transport_first_queries)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
