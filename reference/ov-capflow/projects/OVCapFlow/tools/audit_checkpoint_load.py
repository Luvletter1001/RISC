import argparse
import json
import re
from pathlib import Path

import torch
from mmengine import Config
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import HOOKS, MODELS
from mmrotate.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def unwrap_state_dict(checkpoint):
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    return {
        re.sub(r'^module\.', '', key): value
        for key, value in state.items()
    }


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if 'custom_imports' in cfg:
        import_modules_from_strings(**cfg.custom_imports)
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    incompatible = model.load_state_dict(
        unwrap_state_dict(checkpoint), strict=False)

    layer_cfg = cfg.model.decoder.layer_cfg
    semantic_enabled = bool(layer_cfg.enable_semantic_fusion)
    density_enabled = bool(layer_cfg.enable_density_capacity)
    null_enabled = bool(cfg.model.decoder.enable_null_reservoir)
    allowed_missing = []
    if semantic_enabled:
        allowed_missing.append(re.compile(
            r'^decoder\.layers\.\d+\.semantic_fusion\.'))
    if density_enabled:
        allowed_missing.append(re.compile(
            r'^decoder\.layers\.\d+\.density_capacity\.'))
    if null_enabled:
        allowed_missing.append(re.compile(r'^decoder\.null_reservoir\.'))
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not any(pattern.search(key) for pattern in allowed_missing)
    ]

    trainable = []
    for hook_cfg in cfg.get('custom_hooks', []):
        if hook_cfg.get('type') == 'FreezeExceptHook':
            trainable = HOOKS.build(hook_cfg).apply(model)

    report = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
        'invalid_missing_keys': invalid_missing,
        'enable_semantic_fusion': semantic_enabled,
        'enable_density_capacity': density_enabled,
        'enable_null_reservoir': null_enabled,
        'balanced_enabled': bool(cfg.model.bbox_head.balanced_cfg.enabled),
        'trainable_parameters': trainable,
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    if invalid_missing or incompatible.unexpected_keys:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
