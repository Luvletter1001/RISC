#!/usr/bin/env python3
"""OpenRSD OVD feature/text/logit hook diagnostic.

The first version of this experiment only compared final predictions.  This
version loads the real OpenRSD eval configs, attaches forward hooks to the
model, and records aggregate internal statistics for backbone/FPN features,
support text mappings, alignment logits, fusion logits, objectness/class
scores when available, and bbox outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


CSV_FIELDS = [
    'image_id', 'angle', 'head', 'prompt_family', 'class_name', 'layer',
    'feature_cosine', 'feature_l2', 'text_embedding_norm',
    'image_embedding_norm', 'text_image_similarity',
    'text_image_similarity_drop', 'alignment_logit_top1',
    'fusion_logit_top1', 'logit_kl', 'top1_consistent',
    'top5_consistent', 'objectness', 'objectness_drop', 'class_score',
    'class_score_drop', 'matched_iou', 'center_distance', 'box_angle_diff',
    'semantic_consistent', 'hook_status'
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def load_pickle(path: Path) -> Any:
    with path.open('rb') as f:
        return pickle.load(f)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'tensor'):
        value = value.tensor
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    if hasattr(value, 'cpu'):
        return value.cpu().numpy()
    return np.asarray(value)


def normalize_img_id(img_id: Any) -> str:
    text = str(img_id)
    if text.startswith('angle_') and '__' in text:
        return text.split('__', 1)[1]
    return text


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _fmt_none(row: dict[str, Any]) -> dict[str, Any]:
    return {k: ('' if v is None else v) for k, v in row.items()}


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    n = min(a.size, b.size)
    if n == 0:
        return None
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def _l2(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    n = min(a.size, b.size)
    if n == 0:
        return None
    return float(np.linalg.norm(a[:n].astype(np.float64) - b[:n].astype(np.float64)))


def _tensor_vec(tensor: Any, max_dim: int = 512) -> np.ndarray | None:
    try:
        import torch
    except Exception:  # noqa: BLE001
        torch = None
    if torch is None or not torch.is_tensor(tensor):
        return None
    if tensor.numel() == 0 or not torch.is_floating_point(tensor):
        return None
    arr = tensor.detach().float()
    if arr.ndim >= 4:
        vec = arr.mean(dim=(0, 2, 3))
    elif arr.ndim == 3:
        vec = arr.mean(dim=(0, 1))
    elif arr.ndim == 2:
        vec = arr.mean(dim=0)
    else:
        vec = arr.flatten()
    vec = vec[:max_dim].cpu().numpy().astype(np.float64)
    if vec.size == 0:
        return None
    return vec


def _iter_tensors(value: Any) -> list[Any]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        torch = None
    tensors = []
    if torch is not None and torch.is_tensor(value):
        tensors.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(_iter_tensors(item))
    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(_iter_tensors(item))
    return tensors


def _max_sigmoid(value: Any) -> float | None:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    values = []
    for tensor in _iter_tensors(value):
        if tensor.numel() and torch.is_floating_point(tensor):
            values.append(float(tensor.detach().float().sigmoid().max().cpu()))
    return max(values) if values else None


def _mean_abs(value: Any) -> float | None:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    values = []
    for tensor in _iter_tensors(value):
        if tensor.numel() and torch.is_floating_point(tensor):
            values.append(float(tensor.detach().float().abs().mean().cpu()))
    return float(np.mean(values)) if values else None


@dataclass
class StatBucket:
    sum_vec: np.ndarray | None = None
    count: int = 0
    extras: dict[str, list[float]] = field(default_factory=dict)

    def add_vec(self, vec: np.ndarray | None) -> None:
        if vec is None:
            return
        if self.sum_vec is None:
            self.sum_vec = np.zeros_like(vec, dtype=np.float64)
        n = min(self.sum_vec.size, vec.size)
        self.sum_vec[:n] += vec[:n]
        self.count += 1

    def add_extra(self, name: str, value: Any) -> None:
        val = _safe_float(value)
        if val is not None:
            self.extras.setdefault(name, []).append(val)

    def mean_vec(self) -> np.ndarray | None:
        if self.sum_vec is None or self.count <= 0:
            return None
        return self.sum_vec / max(self.count, 1)

    def mean_extra(self, name: str) -> float | None:
        values = self.extras.get(name, [])
        return float(np.mean(values)) if values else None


class HookCollector:
    def __init__(self, head: str):
        self.head = head
        self.buckets: dict[str, StatBucket] = {}
        self.errors: list[str] = []

    def bucket(self, layer: str) -> StatBucket:
        return self.buckets.setdefault(layer, StatBucket())

    def add_tensor(self, layer: str, tensor: Any, **extras: Any) -> None:
        bucket = self.bucket(layer)
        bucket.add_vec(_tensor_vec(tensor))
        for key, value in extras.items():
            bucket.add_extra(key, value)

    def hook_backbone(self, _module: Any, _inputs: Any, output: Any) -> None:
        try:
            tensors = _iter_tensors(output)
            if tensors:
                self.add_tensor('backbone_last_feature', tensors[-1],
                                image_embedding_norm=_mean_abs(tensors[-1]))
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f'backbone hook failed: {exc}')

    def hook_neck(self, _module: Any, _inputs: Any, output: Any) -> None:
        try:
            tensors = _iter_tensors(output)
            for idx, tensor in enumerate(tensors[:5]):
                self.add_tensor(f'neck_P{idx + 3}', tensor,
                                image_embedding_norm=_mean_abs(tensor))
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f'neck hook failed: {exc}')

    def hook_text_mapping(self, _module: Any, _inputs: Any, output: Any) -> None:
        try:
            tensors = _iter_tensors(output)
            if tensors:
                tensor = tensors[0]
                self.add_tensor('text_support_mapping', tensor,
                                text_embedding_norm=_mean_abs(tensor))
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f'text mapping hook failed: {exc}')

    def hook_visual_mapping(self, _module: Any, _inputs: Any, output: Any) -> None:
        try:
            tensors = _iter_tensors(output)
            if tensors:
                tensor = tensors[0]
                self.add_tensor('image_prompt_embedding', tensor,
                                image_embedding_norm=_mean_abs(tensor))
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f'visual mapping hook failed: {exc}')

    def hook_head(self, name: str):
        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            try:
                if not isinstance(output, (list, tuple)) or len(output) < 2:
                    self.add_tensor(name, output)
                    return
                cls_scores = output[0]
                bbox_preds = output[1] if len(output) > 1 else None
                angle_preds = output[2] if len(output) > 2 else None
                pred_embeds = output[3] if len(output) > 3 else None
                support_feats = output[-1] if len(output) > 4 else None
                logit_layer = 'alignment_logits' if name == 'bbox_head' else 'fusion_logits'
                self.add_tensor(
                    logit_layer,
                    cls_scores[0] if isinstance(cls_scores, (list, tuple)) and cls_scores else cls_scores,
                    alignment_logit_top1=_max_sigmoid(cls_scores) if name == 'bbox_head' else None,
                    fusion_logit_top1=_max_sigmoid(cls_scores) if name != 'bbox_head' else None,
                    class_score=_max_sigmoid(cls_scores),
                    objectness=None,
                )
                self.add_tensor('bbox_output', bbox_preds,
                                center_distance=_mean_abs(bbox_preds))
                self.add_tensor('bbox_angle_output', angle_preds,
                                box_angle_diff=_mean_abs(angle_preds))
                self.add_tensor('image_prompt_embedding', pred_embeds,
                                image_embedding_norm=_mean_abs(pred_embeds))
                self.add_tensor('text_support_after_head', support_feats,
                                text_embedding_norm=_mean_abs(support_feats))
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f'{name} hook failed: {exc}')
        return _hook


def topk_for_sample(sample: dict[str, Any], class_names: list[str], k: int = 5) -> list[dict[str, Any]]:
    inst = sample.get('pred_instances', {})
    labels = to_numpy(inst.get('labels', [])).astype(int)
    scores = to_numpy(inst.get('scores', [])).astype(float)
    boxes = to_numpy(inst.get('bboxes', [])).astype(float)
    order = np.argsort(-scores)[:k] if scores.size else []
    out = []
    for idx in order:
        label = int(labels[idx])
        out.append({
            'label': label,
            'class_name': class_names[label] if 0 <= label < len(class_names) else f'INVALID_{label}',
            'score': float(scores[idx]),
            'bbox': boxes[idx].tolist() if boxes.size else [],
        })
    return out


def support_stats(support_dir: Path, prompt_family: str) -> dict[str, dict[str, float | None]]:
    pkl = support_dir / f'{prompt_family}.pkl'
    if not pkl.exists() and prompt_family.startswith('ovd3_'):
        pkl = support_dir / (prompt_family.replace('ovd3_', '', 1) + '.pkl')
    if not pkl.exists():
        return {}
    support = load_pickle(pkl)
    out = {}
    for name, payload in support.items():
        if not isinstance(payload, dict):
            continue
        text = payload.get('text_embeds')
        visual = payload.get('visual_embeds')
        text_norm = float(np.linalg.norm(np.asarray(text))) if text is not None else None
        image_norm = float(np.linalg.norm(np.asarray(visual))) if visual is not None else None
        sim = None
        if text is not None and visual is not None:
            text_vec = np.asarray(text).reshape(-1)
            vis = np.asarray(visual)
            vis_vec = vis.reshape(-1, vis.shape[-1]).mean(axis=0)
            if text_vec.shape == vis_vec.shape:
                sim = float(np.dot(text_vec, vis_vec) / (np.linalg.norm(text_vec) * np.linalg.norm(vis_vec) + 1e-8))
        out[name] = {
            'text_embedding_norm': text_norm,
            'image_embedding_norm': image_norm,
            'text_image_similarity': sim,
        }
    return out


def find_eval_config(existing_work_dir: Path, prompt_family: str, head: str, angle: str) -> Path | None:
    candidates = [
        existing_work_dir / f'exp_ovd3/ovd3_{prompt_family}/{head}/angle_{angle}/eval_config.py',
        existing_work_dir / f'exp_ovd2/{prompt_family}/{head}/angle_{angle}/eval_config.py',
        existing_work_dir / f'exp_ovd1/{prompt_family}/{head}/angle_{angle}/eval_config.py',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def setup_openrsd(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / 'tools'))
    try:
        from openrsd_env import preload_installed_mmengine
        preload_installed_mmengine()
    except Exception:
        pass
    from mmdet.utils import register_all_modules as register_mmdet
    from mmrotate.utils import register_all_modules as register_mmrotate
    register_mmdet(init_default_scope=False)
    register_mmrotate(init_default_scope=False)


def _parse_dataset_indices(raw: str | None, max_images: int) -> list[int] | None:
    if raw:
        path = Path(raw)
        if path.exists():
            text = path.read_text(encoding='utf-8')
        else:
            text = raw
        indices = [int(item.strip()) for item in text.replace('\n', ',').split(',')
                   if item.strip()]
        return indices[:max_images] if max_images > 0 else indices
    if max_images > 0:
        return list(range(max_images))
    return None


def _assign_dataset_indices(dataset_cfg: Any, indices: list[int] | None) -> None:
    if not indices:
        return
    if isinstance(dataset_cfg, dict):
        if dataset_cfg.get('type') == 'mmdet.ConcatDataset':
            for item in dataset_cfg.get('datasets', []):
                _assign_dataset_indices(item, indices)
        else:
            dataset_cfg['indices'] = list(indices)


def run_live_hook(args: argparse.Namespace, head: str, angle: str, cfg_path: Path) -> tuple[list[dict[str, Any]], list[str], dict[str, np.ndarray]]:
    setup_openrsd(args.repo_root)
    from mmengine.config import Config
    from mmengine.registry import RUNNERS
    from mmengine.runner import Runner

    cfg = Config.fromfile(str(cfg_path))
    cfg.work_dir = str(args.live_work_dir / head / f'angle_{angle}')
    cfg.launcher = 'none'
    if args.checkpoint:
        cfg.load_from = str(args.checkpoint)
    if 'custom_hooks' in cfg:
        cfg.custom_hooks = [
            hook for hook in cfg.custom_hooks
            if hook.get('type') != 'EMAHook'
        ]
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    dataset_indices = _parse_dataset_indices(args.dataset_indices, args.max_live_images)
    _assign_dataset_indices(cfg.test_dataloader.dataset, dataset_indices)
    os.makedirs(cfg.work_dir, exist_ok=True)

    runner = Runner.from_cfg(cfg) if 'runner_type' not in cfg else RUNNERS.build(cfg)
    model = runner.model
    model.eval()
    collector = HookCollector(head=head)
    handles = []
    for module_name, hook_fn in [
        ('backbone', collector.hook_backbone),
        ('neck', collector.hook_neck),
        ('text_support_mapping', collector.hook_text_mapping),
        ('visual_support_mapping', collector.hook_visual_mapping),
        ('bbox_head', collector.hook_head('bbox_head')),
        ('aux_bbox_head', collector.hook_head('aux_bbox_head')),
    ]:
        module = getattr(model, module_name, None)
        if module is not None:
            handles.append(module.register_forward_hook(hook_fn))

    errors: list[str] = []
    try:
        import torch
        runner.load_or_resume()
        dataloader = runner.build_dataloader(
            cfg.test_dataloader,
            seed=getattr(runner, 'seed', None),
            diff_rank_seed=getattr(runner, '_randomness_cfg', {}).get('diff_rank_seed', False)
            if hasattr(runner, '_randomness_cfg') else False,
        )
        seen = 0
        with torch.no_grad():
            for data_batch in dataloader:
                model.test_step(data_batch)
                seen += 1
                if seen >= args.max_live_images:
                    break
        metrics = {'manual_forward_batches': seen, 'evaluator_skipped': True}
        (args.live_work_dir / head / f'angle_{angle}' / 'hook_metrics.json').write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    except Exception as exc:  # noqa: BLE001
        errors.append(f'runner.test failed for {head} angle {angle}: {repr(exc)}')
    finally:
        for handle in handles:
            handle.remove()

    errors.extend(collector.errors)
    rows = []
    vectors = {}
    for layer, bucket in sorted(collector.buckets.items()):
        vec = bucket.mean_vec()
        vectors[layer] = vec if vec is not None else np.zeros(0, dtype=np.float64)
        rows.append({
            'image_id': 'AGGREGATE',
            'angle': angle,
            'head': head,
            'prompt_family': args.prompt_family,
            'class_name': 'ALL',
            'layer': layer,
            'feature_cosine': None,
            'feature_l2': None,
            'text_embedding_norm': bucket.mean_extra('text_embedding_norm'),
            'image_embedding_norm': bucket.mean_extra('image_embedding_norm'),
            'text_image_similarity': None,
            'text_image_similarity_drop': None,
            'alignment_logit_top1': bucket.mean_extra('alignment_logit_top1'),
            'fusion_logit_top1': bucket.mean_extra('fusion_logit_top1'),
            'logit_kl': None,
            'top1_consistent': None,
            'top5_consistent': None,
            'objectness': bucket.mean_extra('objectness'),
            'objectness_drop': None,
            'class_score': bucket.mean_extra('class_score'),
            'class_score_drop': None,
            'matched_iou': None,
            'center_distance': bucket.mean_extra('center_distance'),
            'box_angle_diff': bucket.mean_extra('box_angle_diff'),
            'semantic_consistent': None,
            'hook_status': 'HOOK_OK' if not errors else 'HOOK_PARTIAL',
        })
    if not rows:
        rows.append({
            'image_id': 'AGGREGATE',
            'angle': angle,
            'head': head,
            'prompt_family': args.prompt_family,
            'class_name': 'ALL',
            'layer': 'live_hook',
            'hook_status': 'HOOK_FAILED',
        })
    return rows, errors, vectors


def choose_prediction_rows(existing_work_dir: Path, prompt_family: str, heads: list[str], angles: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    notes = []
    reports = [
        load_json(existing_work_dir / 'exp_ovd3/exp_ovd3_results.json'),
        load_json(existing_work_dir / 'exp_ovd2/exp_ovd2_results.json'),
    ]
    rows = []
    for report in reports:
        for row in report.get('rows', []):
            if row.get('status') != 'OK':
                continue
            if row.get('angle') not in angles:
                continue
            if row.get('head', 'alignment') not in heads:
                continue
            key = row.get('prompt_key', '')
            if prompt_family and prompt_family not in {key, key.replace('ovd3_', '')}:
                continue
            rows.append(row)
        if rows:
            notes.append(f'selected prediction rows from report prompt={prompt_family or "any"}')
            return rows, notes
    notes.append('no matching prediction rows found')
    return rows, notes


def append_prediction_proxy(args: argparse.Namespace,
                            angles: list[str],
                            heads: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows, notes = choose_prediction_rows(args.existing_work_dir, args.prompt_family, heads, angles)
    class_names = []
    if rows:
        class_names = list(rows[0].get('per_class_ap50', {}).keys())
    if not class_names:
        class_names = ['plane', 'bridge', 'ship', 'small-vehicle', 'large-vehicle',
                       'storage-tank', 'harbor', 'roundabout', 'helicopter']
    support = support_stats(args.existing_work_dir / 'prompt_support', args.prompt_family)
    by_head_angle = {(r.get('head', 'alignment'), r.get('angle')): r for r in rows}
    out_rows = []
    for head in heads:
        canon = by_head_angle.get((head, '000')) or next((r for r in rows if r.get('angle') == '000'), None)
        if not canon:
            continue
        canon_samples = {normalize_img_id(s.get('img_id')): s for s in load_pickle(Path(canon['predictions']))[:args.max_images]}
        for angle in angles:
            row = by_head_angle.get((head, angle))
            if not row:
                continue
            samples = {normalize_img_id(s.get('img_id')): s for s in load_pickle(Path(row['predictions']))[:args.max_images]}
            for image_id, sample in samples.items():
                current = topk_for_sample(sample, class_names, 5)
                base = topk_for_sample(canon_samples.get(image_id, {}), class_names, 5)
                if not current:
                    continue
                cur0 = current[0]
                base0 = base[0] if base else {}
                class_name = cur0['class_name']
                sstats = support.get(class_name, {})
                score_drop = (float(base0.get('score', 0.0)) - float(cur0['score'])) if base0 else None
                out_rows.append({
                    'image_id': image_id,
                    'angle': angle,
                    'head': head,
                    'prompt_family': row.get('prompt_key', args.prompt_family),
                    'class_name': class_name,
                    'layer': 'prediction_proxy',
                    'feature_cosine': None,
                    'feature_l2': None,
                    'text_embedding_norm': sstats.get('text_embedding_norm'),
                    'image_embedding_norm': sstats.get('image_embedding_norm'),
                    'text_image_similarity': sstats.get('text_image_similarity'),
                    'text_image_similarity_drop': None,
                    'alignment_logit_top1': cur0['score'] if head == 'alignment' else None,
                    'fusion_logit_top1': cur0['score'] if head == 'fusion' else None,
                    'logit_kl': None,
                    'top1_consistent': bool(base and cur0['label'] == base0.get('label')),
                    'top5_consistent': bool(base and cur0['label'] in {item['label'] for item in base}),
                    'objectness': cur0['score'],
                    'objectness_drop': score_drop,
                    'class_score': cur0['score'],
                    'class_score_drop': score_drop,
                    'matched_iou': None,
                    'center_distance': None,
                    'box_angle_diff': None,
                    'semantic_consistent': bool(base and cur0['label'] == base0.get('label')),
                    'hook_status': 'PREDICTION_PROXY',
                })
    return out_rows, notes


def run(args: argparse.Namespace) -> dict[str, Any]:
    angles = [f'{int(a):03d}' for a in args.angles.split(',') if a]
    heads = [h for h in args.heads.split(',') if h]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.live_work_dir.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []
    errors: list[str] = []
    out_rows: list[dict[str, Any]] = []
    vectors: dict[tuple[str, str, str], np.ndarray] = {}
    live_runs = []

    if args.live_hooks:
        for head in heads:
            for angle in angles:
                cfg_path = args.direct_config or find_eval_config(
                    args.existing_work_dir, args.prompt_family, head, angle)
                if cfg_path is None:
                    errors.append(f'missing eval_config for head={head} angle={angle}')
                    continue
                rows, run_errors, run_vectors = run_live_hook(args, head, angle, cfg_path)
                out_rows.extend(rows)
                errors.extend(run_errors)
                live_runs.append({'head': head, 'angle': angle, 'config': str(cfg_path), 'errors': run_errors})
                for layer, vec in run_vectors.items():
                    vectors[(head, angle, layer)] = vec

        for row in out_rows:
            key = (row.get('head'), row.get('angle'), row.get('layer'))
            base_key = (row.get('head'), '000', row.get('layer'))
            row['feature_cosine'] = _cosine(vectors.get(key), vectors.get(base_key))
            row['feature_l2'] = _l2(vectors.get(key), vectors.get(base_key))
            if row.get('angle') == '000':
                row['text_image_similarity_drop'] = 0.0
            elif row.get('text_image_similarity') not in (None, ''):
                row['text_image_similarity_drop'] = None

    if args.include_prediction_proxy:
        proxy_rows, proxy_notes = append_prediction_proxy(args, angles, heads)
        out_rows.extend(proxy_rows)
        notes.extend(proxy_notes)

    with args.out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(_fmt_none({field: row.get(field) for field in CSV_FIELDS}))

    hook_ok_layers = sorted({
        row.get('layer') for row in out_rows
        if row.get('hook_status') in {'HOOK_OK', 'HOOK_PARTIAL'}
    })
    expected = {
        'backbone_last_feature', 'neck_P3', 'neck_P4', 'neck_P5',
        'text_support_mapping', 'alignment_logits', 'fusion_logits',
        'bbox_output', 'bbox_angle_output', 'image_prompt_embedding',
    }
    missing = sorted(expected - set(hook_ok_layers))
    status = 'DONE' if {'backbone_last_feature', 'neck_P3', 'text_support_mapping'} <= set(hook_ok_layers) and not missing else 'PARTIAL'
    if not out_rows:
        status = 'FAILED'
    summary = {
        'status': status,
        'csv': str(args.out_csv),
        'row_count': len(out_rows),
        'notes': notes,
        'live_hook_runs': live_runs,
        'successful_hooks': hook_ok_layers,
        'failed_or_missing_hooks': missing,
        'errors': errors[:200],
        'live_hooks': bool(args.live_hooks),
        'include_prediction_proxy': bool(args.include_prediction_proxy),
    }
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, default=Path('/data1/zcy/OpenRSD'))
    parser.add_argument('--existing-work-dir', type=Path, default=Path('/data1/zcy/OpenRSD/work_dirs/openrsd_ovd_rotation_20260508'))
    parser.add_argument('--prompt-family', default='F3_orientation_aware')
    parser.add_argument('--angles', default='000,030,060,090,120,150,240')
    parser.add_argument('--heads', default='alignment,fusion')
    parser.add_argument('--max-images', type=int, default=512)
    parser.add_argument('--max-live-images', type=int, default=16)
    parser.add_argument(
        '--dataset-indices',
        default=None,
        help='Comma-separated indices or a file containing validation dataset indices.')
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument(
        '--direct-config',
        type=Path,
        default=None,
        help='Optional explicit eval config used instead of searching existing-work-dir.')
    parser.add_argument('--live-work-dir', type=Path, required=True)
    parser.add_argument('--live-hooks', action='store_true')
    parser.add_argument('--include-prediction-proxy', action='store_true')
    parser.add_argument('--out-csv', type=Path, required=True)
    parser.add_argument('--out-json', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == '__main__':
    main()
