#!/usr/bin/env python3
"""Auto-discovery: configs, checkpoints, VLMs, OpenRSD hooks."""
from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any, Dict, List

from M_Tools.rotation_sv_repair.common import (
    CLASSES, DEFAULT_CHECKPOINT, DEFAULT_CONFIG, REPO_ROOT, SMALL, SuiteContext,
    VLM_SEARCH_ROOTS, now_iso, write_csv,
)

VLM_CATALOG = [
    'RemoteCLIP', 'GeoRSCLIP', 'RS5M', 'RotCLIP', 'RoRoCLIP', 'OpenCLIP', 'CLIP',
    'DINO', 'DINOv2', 'LAE-DINO',
]

VLM_PATTERNS = {
    'RemoteCLIP': [r'remoteclip', r'remote.?clip'],
    'GeoRSCLIP': [r'georsclip', r'geo.?rs.?clip'],
    'RS5M': [r'rs5m'],
    'RotCLIP': [r'rotclip'],
    'RoRoCLIP': [r'roroclip', r'roro.?clip'],
    'OpenCLIP': [r'open.?clip', r'open_clip'],
    'CLIP': [r'\bclip\b', r'ViT-B/32', r'ViT-L/14'],
    'DINO': [r'\bdino\b(?!v2)'],
    'DINOv2': [r'dinov2', r'dino_v2', r'dino-v2'],
    'LAE-DINO': [r'lae.?dino', r'laedino'],
}

HOOK_TARGETS = [
    ('ContrastiveEmbed', 'bbox_head', 'classification'),
    ('rtm_cls_heads', 'bbox_head.rtm_cls_heads', 'classification'),
    ('bbox_head', 'bbox_head', 'detection'),
    ('alignment_head', 'bbox_head', 'alignment'),
    ('fusion_head', 'bbox_head', 'fusion'),
    ('text_prompt', 'text_support_mapping', 'prompt'),
    ('image_prompt', 'visual_support_mapping', 'prompt'),
    ('support_embedding', 'support_feat_dict', 'embedding'),
    ('class_embedding', 'normalized_class_dict', 'embedding'),
    ('dense_visual', 'bbox_head.forward', 'dense'),
    ('post_nms', 'predict', 'postprocess'),
]


def _glob_roots(roots: List[Path], patterns: List[str], max_hits: int = 5) -> List[Path]:
    """Fast bounded search — avoid full-tree rglob on /data1/zcy."""
    import subprocess
    hits: List[Path] = []
    for root in roots:
        if not root.exists() or len(hits) >= max_hits:
            continue
        for pat in patterns[:2]:
            try:
                proc = subprocess.run(
                    ['find', str(root), '-maxdepth', '4', '-iname', f'*{pat}*',
                     '(', '-name', '*.pth', '-o', '-name', '*.pt', '-o', '-name', '*.bin', '-o',
                     '-name', '*.safetensors', ')', '-type', 'f'],
                    capture_output=True, text=True, timeout=8,
                )
                for line in proc.stdout.splitlines():
                    p = Path(line.strip())
                    if p.is_file() and p.stat().st_size > 1024:
                        hits.append(p.resolve())
                        if len(hits) >= max_hits:
                            return hits
            except (OSError, PermissionError, subprocess.TimeoutExpired):
                continue
    return hits


KNOWN_VLM_HINTS = {
    'RemoteCLIP': ['pretrained/remoteclip'],
    'GeoRSCLIP': ['pretrained/georsclip'],
    'RS5M': ['pretrained/georsclip/ckpt'],
    'LAE-DINO': ['pretrained/lae-dino/checkpoints'],
    'CLIP': ['pretrained/clip', 'checkpoints/clip'],
    'DINOv2': ['pretrained/dinov2'],
    'DINO': ['pretrained/dino'],
    'OpenCLIP': ['pretrained/open_clip'],
    'RotCLIP': ['pretrained/rotclip'],
    'RoRoCLIP': ['pretrained/roroclip'],
}


def _ckpts_under(path: Path, max_hits: int = 3) -> List[Path]:
    if not path.exists():
        return []
    if path.is_file() and path.stat().st_size > 1024:
        return [path.resolve()]
    hits: List[Path] = []
    for ext in ('*.pth', '*.pt', '*.bin', '*.safetensors'):
        for p in sorted(path.rglob(ext)):
            if p.is_file() and p.stat().st_size > 1024:
                hits.append(p.resolve())
                if len(hits) >= max_hits:
                    return hits
    return hits


def scan_vlm_inventory(ctx: SuiteContext) -> List[dict]:
    rows = []
    roots = [ctx.repo_root, ctx.repo_root / 'pretrained', ctx.repo_root / 'checkpoints']
    if ctx.mode == 'full':
        roots.extend([Path('/data1/zcy/OpenRSD/pretrained')])
    for vlm in VLM_CATALOG:
        ckpts: List[Path] = []
        for rel in KNOWN_VLM_HINTS.get(vlm, []):
            ckpts.extend(_ckpts_under(ctx.repo_root / rel, max_hits=3))
        if ctx.mode == 'full':
            pats = VLM_PATTERNS.get(vlm, [vlm.lower()])
            ckpts.extend(_glob_roots(roots, pats, max_hits=2))
        ckpts = list(dict.fromkeys(ckpts))[:3]
        available = bool(ckpts)
        blocked = '' if available else 'BLOCKED_NOT_AVAILABLE'
        teacher_ready = available and vlm in (
            'RemoteCLIP', 'GeoRSCLIP', 'RS5M', 'LAE-DINO', 'OpenCLIP', 'CLIP', 'DINOv2')
        rows.append(dict(
            vlm=vlm,
            available='yes' if available else 'no',
            checkpoint=';'.join(str(c) for c in ckpts) if ckpts else '',
            config='',
            tokenizer_or_processor='open_clip' if vlm in ('RemoteCLIP', 'GeoRSCLIP', 'RS5M') else '',
            embedding_dim='',
            can_encode_image='yes' if teacher_ready and vlm in ('RemoteCLIP', 'GeoRSCLIP', 'RS5M') else (
                'unknown' if available else 'no'),
            can_encode_text='yes' if teacher_ready and vlm in ('RemoteCLIP', 'GeoRSCLIP', 'RS5M') else (
                'unknown' if available else 'no'),
            used_as_teacher='yes' if teacher_ready else 'no',
            blocked_reason=blocked,
        ))
    return rows


def scan_openrsd_hooks(ctx: SuiteContext) -> List[dict]:
    rows = []
    try:
        from mmengine.config import Config
        cfg_path, _, _ = __import__(
            'M_Tools.rotation_sv_repair.common', fromlist=['discover_config_checkpoint']
        ).discover_config_checkpoint(ctx)
        cfg = Config.fromfile(str(cfg_path))
        model_type = cfg.model.get('type', 'unknown')
    except Exception as exc:
        model_type = f'config_error:{exc}'

    for name, path_hint, mtype in HOOK_TARGETS:
        rows.append(dict(
            module_name=name,
            module_type=mtype,
            input_shape='(B,C,H,W) or (N,D)',
            output_shape='(B,C,H,W) or detections',
            requires_grad='frozen_at_inference',
            used_for_baseline='yes' if name in ('bbox_head', 'ContrastiveEmbed', 'post_nms') else 'partial',
            used_for_adapter='yes' if name in ('dense_visual', 'rtm_cls_heads', 'ContrastiveEmbed') else 'no',
            used_for_teacher='yes' if name in ('alignment_head', 'fusion_head', 'text_prompt', 'image_prompt') else 'partial',
            used_for_distill='yes' if name in ('dense_visual', 'alignment_head', 'fusion_head') else 'no',
            path_hint=path_hint,
            model_type=str(model_type),
        ))
    return rows


def discover_assets(ctx: SuiteContext) -> Dict[str, Any]:
    config, ckpt, support = __import__(
        'M_Tools.rotation_sv_repair.common', fromlist=['discover_config_checkpoint']
    ).discover_config_checkpoint(ctx)
    angle_ok = (REPO_ROOT / 'data/DOTA1_1024_500/angle_sweep_val/realistic/angle_000/images').is_dir()
    stems_n = len(__import__(
        'M_Tools.rotation_sv_repair.common', fromlist=['list_angle_sweep_stems']
    ).list_angle_sweep_stems(0, 10))
    return dict(
        config=str(config), config_exists=config.exists(),
        checkpoint=str(ckpt), checkpoint_exists=ckpt.exists(),
        support=str(support), support_exists=support.exists(),
        angle_sweep_available=angle_ok,
        angle_sweep_sample_stems=stems_n,
        small_vehicle_index=SMALL,
        classes=CLASSES,
        git=__import__('M_Tools.rotation_sv_repair.common', fromlist=['git_commit']).git_commit(ctx.repo_root),
    )


def run_inventory(ctx: SuiteContext) -> dict:
    fres = 'fres_002_external_vlm_inventory.md'
    csv_path = ctx.tables_dir / 'ftable_002_external_vlm_inventory.csv'
    if ctx.should_skip_output(ctx.fres_path(fres)) and ctx.should_skip_output(csv_path):
        return dict(status='DONE', skipped=True)
    rows = scan_vlm_inventory(ctx)
    fields = list(rows[0].keys()) if rows else []
    write_csv(csv_path, rows, fields)
    n_avail = sum(1 for r in rows if r['available'] == 'yes')
    status = 'DONE' if rows else 'PARTIAL'
    with open(ctx.fres_path(fres), 'w', encoding='utf-8') as f:
        from M_Tools.rotation_sv_repair.common import write_fres_header
        write_fres_header(f, 'External VLM / Rotation-Robust CLIP Inventory', ctx, status)
        f.write(f'- models scanned: {len(VLM_CATALOG)}\n')
        f.write(f'- locally available: **{n_avail}**\n')
        f.write(f'- csv: `{csv_path}`\n\n')
        f.write('| vlm | available | checkpoint | blocked_reason |\n')
        f.write('|:---|:---|:---|:---|\n')
        for r in rows:
            ck = (r['checkpoint'][:80] + '...') if len(r['checkpoint']) > 80 else r['checkpoint']
            f.write(f"| {r['vlm']} | {r['available']} | `{ck}` | {r['blocked_reason']} |\n")
    ctx.mark_step('inventory', status)
    return dict(status=status, csv=str(csv_path), fres=str(ctx.fres_path(fres)))


def run_hooks(ctx: SuiteContext) -> dict:
    fres = 'fres_003_openrsd_hook_map.md'
    csv_path = ctx.tables_dir / 'ftable_003_openrsd_hook_map.csv'
    if ctx.should_skip_output(ctx.fres_path(fres)) and ctx.should_skip_output(csv_path):
        return dict(status='DONE', skipped=True)
    rows = scan_openrsd_hooks(ctx)
    fields = ['module_name', 'module_type', 'input_shape', 'output_shape', 'requires_grad',
              'used_for_baseline', 'used_for_adapter', 'used_for_teacher', 'used_for_distill',
              'path_hint', 'model_type']
    write_csv(csv_path, rows, fields)
    with open(ctx.fres_path(fres), 'w', encoding='utf-8') as f:
        from M_Tools.rotation_sv_repair.common import write_fres_header
        write_fres_header(f, 'OpenRSD Internal Hook Map', ctx)
        f.write('## Sanity requirements\n\n')
        f.write('1. Repair disabled → identical to baseline.\n')
        f.write('2. adapter alpha=0 → identical to baseline.\n')
        f.write('3. logit bias all zero → identical to baseline.\n')
        f.write(f'4. small_vehicle index verified: **{SMALL}** = `{CLASSES[SMALL]}`\n\n')
        f.write(f'- csv: `{csv_path}`\n\n')
        f.write('| module | type | baseline | adapter | teacher | distill |\n')
        f.write('|:---|:---|:---|:---|:---|:---|\n')
        for r in rows:
            f.write(f"| {r['module_name']} | {r['module_type']} | {r['used_for_baseline']} | "
                    f"{r['used_for_adapter']} | {r['used_for_teacher']} | {r['used_for_distill']} |\n")
    ctx.mark_step('hooks', 'DONE')
    return dict(status='DONE', csv=str(csv_path))


def run_dryrun(ctx: SuiteContext) -> dict:
    fres = 'fres_000_repair_suite_overview.md'
    assets = discover_assets(ctx)
    with open(ctx.fres_path(fres), 'w', encoding='utf-8') as f:
        from M_Tools.rotation_sv_repair.common import write_fres_header
        write_fres_header(f, 'Rotation SV Repair Suite — Overview (Dryrun)', ctx)
        f.write('## Discovered assets\n\n')
        for k, v in assets.items():
            f.write(f'- **{k}:** `{v}`\n')
        f.write('\n## Planned experiments\n\n')
        exps = [
            'splits', 'hooks', 'tests', 'baseline', 'postprocess', 'embedding', 'prompt',
            'adapter', 'orbit', 'ovd_teacher_student', 'official_ap', 'true_sv', 'oracle',
            'stats', 'audit', 'ablation', 'verdict',
        ]
        for e in exps:
            f.write(f'- {e}\n')
        f.write('\n## Output layout\n\n')
        f.write(f'- work_dir: `{ctx.work_dir}`\n')
        f.write(f'- result_dir: `{ctx.result_dir}`\n')
        f.write(f'- logs: `{ctx.logs_dir}`\n')
    ctx.mark_step('dryrun', 'DONE')
    return dict(status='DONE', assets=assets, fres=str(ctx.fres_path(fres)))
