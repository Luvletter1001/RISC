#!/usr/bin/env python
"""Step 0: preflight inventory."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import torch

from tools.verify_sv_attractor_gpu89 import common as C


def run(ctx: C.RunContext) -> dict:
    out = ctx.exp_dir('exp_00_preflight')
    fres = ctx.fres_path('fres_00_preflight_inventory.md')
    status = 'OK'
    err = ''
    inventory = dict(
        git_commit=C.git_commit(ctx.repo_root),
        versions=C.package_versions(),
        config=str(ctx.config),
        checkpoint=str(ctx.checkpoint),
        checkpoint_exists=ctx.checkpoint.exists(),
        support_pkl=str(ctx.support_pkl),
        support_exists=bool(ctx.support_pkl and ctx.support_pkl.exists()),
        support_source=ctx.progress.get('support_source', ''),
    )

    support_keys = []
    class_names_in_pkl = []
    if ctx.support_pkl and ctx.support_pkl.exists():
        with open(ctx.support_pkl, 'rb') as f:
            sp = pickle.load(f)
        support_keys = sorted(sp.keys())[:50]
        if isinstance(sp, dict) and sp:
            first = next(iter(sp.values()))
            if isinstance(first, dict):
                class_names_in_pkl = sorted(sp.keys())

    p0148_img = ctx.repo_root / 'vis' / C.DEFAULT_TILE / 'dataset' / 'images'
    p0148_ann = ctx.repo_root / 'vis' / C.DEFAULT_TILE / 'dataset' / 'annfiles'
    cross_tiles = C.discover_cross_tiles(ctx.repo_root, 30)

    model_info = {}
    head_modules = {}
    can_load = False
    try:
        _, model, _, det_support, name2id, id2name, cfg = C.build_model(ctx)
        can_load = True
        head = model.bbox_head
        model_info['bbox_head_type'] = head.__class__.__name__
        model_info['num_classes_attr'] = getattr(head, 'num_classes', 'N/A')
        model_info['dataset_classes_probe'] = list(det_support.keys())
        model_info['name2id'] = name2id
        checks = [
            'cls_convs', 'rtm_cls', 'rtm_cls_heads', 'rtm_reg', 'rtm_ang']
        named = dict(model.named_modules())
        for key in checks:
            head_modules[key] = any(k.startswith(f'bbox_head.{key}') for k in named)
        aux = 'aux_bbox_head' in named
        head_modules['aux_bbox_head'] = aux
        cls_head_types = []
        if hasattr(head, 'rtm_cls_heads'):
            for i, m in enumerate(head.rtm_cls_heads):
                cls_head_types.append(f'level{i}:{m.__class__.__name__}')
        model_info['rtm_cls_heads_types'] = cls_head_types
        model_info['visual_support_mapping'] = (
            model.visual_support_mapping.__class__.__name__)
        model_info['text_support_mapping'] = (
            model.text_support_mapping.__class__.__name__)
        try:
            from mmengine.config import Config
            c = Config.fromfile(str(ctx.config))
            inventory['cfg_val_support_classes'] = list(
                c.model.get('val_support_classes', []))
            inventory['cfg_val_dataset_flag'] = c.model.get('val_dataset_flag', '')
            sdict = c.model.get('support_feat_dict', {})
            inventory['cfg_support_paths'] = {
                k: str(sdict[k]) for k in sorted(sdict) if 'DOTA' in k}
        except Exception as exc:
            inventory['cfg_parse_note'] = str(exc)
    except Exception as exc:
        status = 'FAILED'
        err = str(exc)
        can_load = False

    inventory['model_load_ok'] = can_load
    inventory['p0148_images_exist'] = p0148_img.is_dir()
    inventory['p0148_ann_exist'] = p0148_ann.is_dir()
    inventory['cross_tile_candidates'] = cross_tiles

    (out / 'preflight_inventory.json').write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False, default=str))

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'Preflight Inventory', ctx, status)
        f.write('## Environment\n\n')
        f.write(f'- git commit: `{inventory["git_commit"]}`\n')
        for k, v in inventory['versions'].items():
            f.write(f'- {k}: `{v}`\n')
        f.write('\n## Paths\n\n')
        f.write(f'1. config: `{inventory["config"]}`\n')
        f.write(f'2. checkpoint: `{inventory["checkpoint"]}` (exists={inventory["checkpoint_exists"]})\n')
        f.write(f'3. support pkl: `{inventory["support_pkl"]}` (exists={inventory["support_exists"]})\n')
        f.write(f'4. support source: `{inventory.get("support_source", "")}`\n')
        f.write(f'5. P0148 images: `{p0148_img}` (exists={inventory["p0148_images_exist"]})\n')
        f.write(f'6. P0148 ann: `{p0148_ann}` (exists={inventory["p0148_ann_exist"]})\n')
        f.write('\n## Support PKL\n\n')
        f.write(f'- sample keys (first 50): `{support_keys[:20]}...` total shown {len(support_keys)}\n')
        f.write(f'- class names count in pkl: `{len(class_names_in_pkl)}`\n')
        if class_names_in_pkl:
            f.write(f'- classes: `{class_names_in_pkl}`\n')
        f.write('\n## Model\n\n')
        if can_load:
            f.write(f'- model load: **OK**\n')
            f.write(f'- bbox_head: `{model_info.get("bbox_head_type")}`\n')
            f.write(f'- num_classes attr: `{model_info.get("num_classes_attr")}`\n')
            f.write(f'- probe DOTA1 classes ({len(model_info.get("dataset_classes_probe", []))}): '
                    f'`{model_info.get("dataset_classes_probe")}`\n')
            f.write(f'- rtm_cls_heads: `{model_info.get("rtm_cls_heads_types")}`\n')
            f.write('\n### Module existence\n\n')
            f.write('| module | exists |\n|---|---|\n')
            for k, v in head_modules.items():
                f.write(f'| bbox_head.{k} | {v} |\n')
            f.write('\n### Config metainfo note\n\n')
            f.write(f'- cfg val_support_classes count: '
                    f'`{len(inventory.get("cfg_val_support_classes", []))}`\n')
            f.write('- **Note**: diagnostics use DOTA1 15-class support via `prepare_support`; '
                    'mapping sanity must confirm index alignment.\n')
        else:
            f.write(f'- model load: **FAILED**\n')
            f.write(f'- error: `{err}`\n')
        f.write('\n## Cross-tile candidates\n\n')
        for t in cross_tiles[:25]:
            f.write(f'- `{t}`\n')
        if len(cross_tiles) > 25:
            f.write(f'- ... and {len(cross_tiles) - 25} more\n')
        f.write('\n## Gate\n\n')
        if not can_load:
            f.write('**STOP**: model cannot load; subsequent experiments are BLOCKED.\n')
            ctx.set_blocked('preflight model load failed')
        else:
            f.write('Preflight passed; proceed to mapping sanity.\n')

    if not can_load:
        ctx.mark_step('preflight', 'FAILED', error=err)
    else:
        ctx.mark_step('preflight', 'OK', fres=str(fres))
    return dict(status=status, fres=str(fres), can_load=can_load, error=err)
