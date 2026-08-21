#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import sys
import time
from collections import Counter, OrderedDict
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path('/data1/zcy/OpenRSD')


def _path_matches(path_item, target):
    if path_item == '':
        path_item = os.getcwd()
    return os.path.abspath(path_item) == str(target)


# Match SimpleRun/step1_inference.py: prefer installed mmengine/mmdet packages,
# then add the project for custom M_AD modules.
sys.path = [
    p for p in sys.path
    if not p.startswith('/home/zcy/.local/')
    and not _path_matches(p, PROJECT_ROOT)
]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / 'SimpleRun/.mplconfig'))

import torch
import torch.nn.functional as F
from mmcv.ops.nms import nms_rotated
from mmengine.config import Config
from mmengine.registry import DATA_SAMPLERS, DATASETS, FUNCTIONS, RUNNERS
from mmengine.runner import Runner
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmrotate.utils import register_all_modules
from torch.utils.data import DataLoader

from ctlib.rbox import obb2poly
from commonlibs.common_tools import pklload, pklsave


DOTA1_CLASSES = [
    'baseball-diamond', 'basketball-court', 'bridge', 'ground-track-field',
    'harbor', 'helicopter', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank',
    'swimming-pool', 'tennis-court',
]

DEFAULT_SUPPORT_CANDIDATES = [
    PROJECT_ROOT / 'data/DOTA_800_600/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
    PROJECT_ROOT / 'data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
    PROJECT_ROOT / 'data/DOTAV2/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Probe OpenRSD stage outputs on rotated images.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--angles', nargs='*', type=int, default=None)
    parser.add_argument('--angle-step', type=int, default=None)
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--support-shot', type=int, default=8)
    parser.add_argument(
        '--support-type',
        choices=['visual', 'text', 'random'],
        default='visual',
        help='Use visual by default to keep logits comparable across angles.')
    parser.add_argument('--support-feat', default='')
    parser.add_argument(
        '--normalized-class-dict',
        default='data/normalized_class_dict.pkl')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-spatial-size', type=int, default=32)
    parser.add_argument(
        '--result-md-dir',
        default='resultmd/exp_rotation_stage_probe_P0148')
    return parser.parse_args()


def angle_from_name(path):
    match = re.search(r'_rot(\d{3})', Path(path).stem)
    if not match:
        return None
    return int(match.group(1))


def selected_angles(args, image_dir):
    if args.angles is not None and len(args.angles) > 0:
        return sorted({int(a) % 360 for a in args.angles})
    if args.angle_step:
        return list(range(0, 360, args.angle_step))
    found = []
    for path in sorted(Path(image_dir).glob('*_rot*.jpg')):
        angle = angle_from_name(path)
        if angle is not None:
            found.append(angle)
    return sorted(set(found))


def resolve_support_path(user_path):
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p, None
        raise FileNotFoundError(f'support feature file not found: {p}')
    for p in DEFAULT_SUPPORT_CANDIDATES:
        if p.exists():
            warning = None
            if p != DEFAULT_SUPPORT_CANDIDATES[0]:
                warning = (
                    f'default step1 support path missing; using fallback {p}')
            return p, warning
    raise FileNotFoundError(
        'no support feature file found in known DOTA candidates')


def ensure_minimal_neg_support(path):
    path = Path(path)
    if not path.exists():
        pklsave({'neg_dict': {'Data1_DOTA1': {c: [] for c in DOTA1_CLASSES}}},
                str(path))
    return str(path)


def setup_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cfg(args, support_path):
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    cfg.work_dir = str(Path(args.out_dir) / 'runner_work_dir')
    cfg.load_from = args.checkpoint
    cfg.model.support_feat_dict = {'Data1_DOTA1': str(support_path)}
    cfg.model.val_support_classes = DOTA1_CLASSES
    cfg.model.val_dataset_flag = 'Data1_DOTA1'
    cfg.model.neg_support_data = ensure_minimal_neg_support(
        PROJECT_ROOT / 'SimpleRun/generated_annfiles/Neg_supports_dota1_minimal.pkl')
    cfg.model.normalized_class_dict = args.normalized_class_dict
    cfg.model.pca_meta_pth = './data/7_25_pca_meta_DINOv2_256.pkl'
    cfg.model.val_using_aux = False
    return cfg


def build_runner_model(args, support_path):
    cfg = build_cfg(args, support_path)
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.call_hook('before_run')
    runner.load_or_resume()
    model = runner.model
    model.eval()
    return runner, model, cfg


def build_dataloader(args, angles):
    image_dir = Path(args.image_dir).resolve()
    dataset_root = image_dir.parent
    ann_dir = dataset_root / 'annfiles'
    if not ann_dir.is_dir():
        raise FileNotFoundError(f'annfiles directory not found: {ann_dir}')

    from M_AD.datasets.transforms.loading import LoadAnnotationsOnline  # noqa: F401
    from M_AD.datasets.transforms.formatting import PackDetInputsMM  # noqa: F401
    from M_AD.datasets.transforms.transforms import ConvertBoxTypeSafe  # noqa: F401

    test_pipeline = [
        dict(type='mmdet.LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(type='mmrotate.LoadAnnotationsOnline', with_bbox=True, box_type='qbox'),
        dict(type='mmrotate.ConvertBoxTypeSafe',
             box_type_mapping=dict(gt_bboxes='rbox')),
        dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
        dict(type='mmdet.Pad', size=(1024, 1024),
             pad_val=dict(img=(114, 114, 114))),
        dict(
            type='mmrotate.PackDetInputsMM',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor')),
    ]
    test_dataset = dict(
        type='mmrotate.DOTADatasetOnline',
        data_root=str(dataset_root),
        ann_file='annfiles',
        data_prefix=dict(img_path='images'),
        img_shape=(1024, 1024),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=test_pipeline)
    dataset = DATASETS.build(test_dataset)
    sampler = DATA_SAMPLERS.build(
        dict(type='DefaultSampler', shuffle=False),
        default_args=dict(dataset=dataset, seed=2024))
    collate_fn = FUNCTIONS.get('pseudo_collate')
    loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=False,
        drop_last=False,
        collate_fn=partial(collate_fn))
    return loader


def prepare_support(args, device):
    norm_cls_map = pklload(args.normalized_class_dict)
    support_path, support_warning = resolve_support_path(args.support_feat)
    support_data = pklload(str(support_path))
    uni_support_data = OrderedDict()
    for raw_name, info in support_data.items():
        # The legacy /data/DOTA_800_600 support uses names that need
        # normalized_class_dict. The local DOTA1_1024 support is already in the
        # lower-case DOTA1 prompt vocabulary used by this diagnostic.
        normed = raw_name if raw_name in DOTA1_CLASSES else norm_cls_map.get(
            raw_name, raw_name)
        if normed not in DOTA1_CLASSES:
            continue
        item = {}
        for key in ['visual_embeds', 'text_embeds']:
            arr = np.asarray(info[key])
            if len(arr) <= args.support_shot:
                arr = np.concatenate([arr for _ in range(args.support_shot + 1)])
            item[key] = arr
        uni_support_data[normed] = item
    missing = [c for c in DOTA1_CLASSES if c not in uni_support_data]
    if missing:
        raise RuntimeError(f'missing DOTA1 support classes: {missing}')
    det_support_data = OrderedDict((c, uni_support_data[c]) for c in DOTA1_CLASSES)
    name2id = {name: idx for idx, name in enumerate(det_support_data.keys())}
    id2name = {idx: name for name, idx in name2id.items()}
    return support_path, support_warning, det_support_data, name2id, id2name


def iter_tensors(obj, prefix='out'):
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            yield from iter_tensors(item, f'{prefix}_{i}')
    elif isinstance(obj, dict):
        for key, item in obj.items():
            yield from iter_tensors(item, f'{prefix}_{key}')


def safe_key(text):
    return re.sub(r'[^0-9A-Za-z_]+', '_', text).strip('_')


def tensor_scalar_stats(tensor):
    t = tensor.detach().float().cpu()
    if t.numel() == 0:
        return dict(shape=list(t.shape), mean=None, std=None, min=None,
                    max=None, l2_norm=0.0)
    return dict(
        shape=list(t.shape),
        mean=float(t.mean().item()),
        std=float(t.std(unbiased=False).item()),
        min=float(t.min().item()),
        max=float(t.max().item()),
        l2_norm=float(torch.linalg.vector_norm(t).item()))


def vector_and_map(tensor, max_spatial_size):
    t = tensor.detach().float().cpu()
    arrays = {}
    if t.ndim >= 4:
        x = t[0]
        arrays['gap'] = x.mean(dim=tuple(range(1, x.ndim))).numpy()
        if x.ndim == 3:
            fmap = x.unsqueeze(0)
            if max(fmap.shape[-2:]) > max_spatial_size:
                fmap = F.interpolate(
                    fmap, size=(max_spatial_size, max_spatial_size),
                    mode='bilinear', align_corners=False)
            arrays['fmap'] = fmap[0].numpy()
            arrays['heatmap'] = fmap[0].mean(dim=0).numpy()
    elif t.ndim == 3:
        arrays['gap'] = t[0].mean(dim=0).numpy()
    elif t.ndim == 2:
        arrays['gap'] = t.mean(dim=0).numpy()
    elif t.ndim == 1:
        arrays['gap'] = t.numpy()
    return arrays


class HookRecorder:
    def __init__(self, max_spatial_size):
        self.max_spatial_size = max_spatial_size
        self.records = OrderedDict()
        self.handles = []

    def clear(self):
        self.records = OrderedDict()

    def add_record(self, stage, module_name, value):
        recs = []
        arrays = {}
        for tensor_path, tensor in iter_tensors(value):
            stat = tensor_scalar_stats(tensor)
            recs.append(dict(stage=stage, module=module_name,
                             tensor_path=tensor_path, **stat))
            prefix = safe_key(f'{stage}__{module_name}__{tensor_path}')
            for arr_name, arr in vector_and_map(
                    tensor, self.max_spatial_size).items():
                arrays[f'{prefix}__{arr_name}'] = arr
        self.records[f'{stage}:{module_name}'] = dict(stats=recs, arrays=arrays)

    def hook(self, stage, module_name):
        def _hook(module, inputs, output):
            self.add_record(stage, module_name, output)
        return _hook

    def pre_hook(self, stage, module_name):
        def _hook(module, inputs):
            if inputs:
                self.add_record(stage, module_name, inputs[0])
        return _hook

    def register(self, model):
        named = dict(model.named_modules())
        hooks = []
        for stage, name in [
                ('backbone', 'backbone'),
                ('neck', 'neck'),
                ('head_output', 'bbox_head')]:
            if name in named:
                hooks.append((stage, name, named[name], False))
        if 'bbox_head' in named:
            hooks.append(('head_input', 'bbox_head', named['bbox_head'], True))
        for name, module in named.items():
            if re.match(r'bbox_head\.cls_convs\.\d+\.\d+$', name):
                hooks.append(('head_cls_branch', name, module, False))
            elif re.match(r'bbox_head\.rtm_cls\.\d+$', name):
                hooks.append(('head_pred_embed', name, module, False))
            elif re.match(r'bbox_head\.rtm_cls_heads\.\d+$', name):
                hooks.append(('cls_logits', name, module, False))
        for stage, name, module, is_pre in hooks:
            if is_pre:
                self.handles.append(
                    module.register_forward_pre_hook(self.pre_hook(stage, name)))
            else:
                self.handles.append(module.register_forward_hook(
                    self.hook(stage, name)))
        return hooks

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def write_module_inventory(model, hooks, out_dir, result_md_dir):
    rows = []
    hooked = {name: stage for stage, name, _, _ in hooks}
    for name, module in model.named_modules():
        rows.append(dict(
            name=name or '<root>',
            type=module.__class__.__name__,
            hooked_stage=hooked.get(name, '')))
    csv_path = Path(out_dir) / 'module_inventory.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'type', 'hooked_stage'])
        writer.writeheader()
        writer.writerows(rows)
    result_md_dir = Path(result_md_dir)
    result_md_dir.mkdir(parents=True, exist_ok=True)
    md_path = result_md_dir / 'fres_module_inventory.md'
    with open(md_path, 'w') as f:
        f.write('# Rotation Stage Probe Module Inventory\n\n')
        f.write(f'- generated_at: `{time.strftime("%Y-%m-%d %H:%M:%S")}`\n')
        f.write(f'- module_count: `{len(rows)}`\n')
        f.write(f'- hooked_module_count: `{len(hooks)}`\n\n')
        f.write('## Hooked Modules\n\n')
        f.write('| stage | module | type |\n|---|---|---|\n')
        for stage, name, module, is_pre in hooks:
            suffix = ' pre-hook' if is_pre else ''
            f.write(f'| {stage}{suffix} | `{name}` | `{module.__class__.__name__}` |\n')
        f.write('\n## Candidate cls / head Modules\n\n')
        f.write('| module | type | hooked_stage |\n|---|---|---|\n')
        for row in rows:
            lname = row['name'].lower()
            if 'bbox_head' in lname and (
                    'cls' in lname or 'rtm' in lname or row['hooked_stage']):
                f.write(
                    f"| `{row['name']}` | `{row['type']}` | `{row['hooked_stage']}` |\n")
    return csv_path, md_path


def get_pred_results(results, id2name, iou_thr=0.5, score_thr=0.3):
    outputs = []
    for sample in results:
        pred_boxes = sample.pred_instances.bboxes.detach()
        pred_labels = sample.pred_instances.labels.detach()
        pred_scores = sample.pred_instances.scores.detach()
        if len(pred_boxes) > 0:
            _, keep_inds = nms_rotated(pred_boxes, pred_scores,
                                       iou_threshold=iou_thr)
            pred_boxes = pred_boxes[keep_inds]
            pred_scores = pred_scores[keep_inds]
            pred_labels = pred_labels[keep_inds]
            keep = pred_scores >= score_thr
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]
        texts = [id2name[int(x)] for x in pred_labels]
        polys = obb2poly(pred_boxes.detach().cpu()).numpy() if len(pred_boxes) else []
        outputs.append(dict(
            bboxes=pred_boxes.detach().cpu().numpy().tolist(),
            polys=np.asarray(polys).tolist(),
            scores=pred_scores.detach().cpu().numpy().tolist(),
            labels=pred_labels.detach().cpu().numpy().astype(int).tolist(),
            class_names=texts))
    return outputs


def summarize_cls_arrays(angle, recorder, class_names, arrays_out):
    rows = []
    small_idx = class_names.index('small-vehicle')
    for key, rec in recorder.records.items():
        if not key.startswith('cls_logits:'):
            continue
        module = key.split(':', 1)[1]
        level_match = re.search(r'rtm_cls_heads\.(\d+)', module)
        level = int(level_match.group(1)) if level_match else -1
        for tensor_path, tensor in iter_tensors(
                [r for r in []], prefix='none'):
            pass
        # Re-read the original arrays through stats keys is not possible here;
        # cls logits are also present in recorder arrays as gap vectors.
        for arr_key, arr in rec['arrays'].items():
            if not arr_key.endswith('__gap'):
                continue
            vec = np.asarray(arr).reshape(-1)
            if len(vec) < len(class_names):
                continue
            logits = vec[:len(class_names)]
            scores = 1.0 / (1.0 + np.exp(-logits))
            prob = scores / max(float(scores.sum()), 1.0e-12)
            order = np.argsort(scores)[::-1]
            arrays_out[f'cls_level_{level}_score_mean'] = scores
            rows.append(dict(
                angle=angle,
                level=level,
                module=module,
                top1_class=class_names[int(order[0])],
                top1_score=float(scores[order[0]]),
                top2_class=class_names[int(order[1])] if len(order) > 1 else '',
                top2_score=float(scores[order[1]]) if len(order) > 1 else 0.0,
                margin=float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0,
                top5=';'.join(
                    f'{class_names[int(i)]}:{float(scores[i]):.6f}'
                    for i in order[:5]),
                small_vehicle_score=float(scores[small_idx]),
                entropy=float(-(prob * np.log(prob + 1.0e-12)).sum())))
    if rows:
        mean_scores = np.mean(
            [arrays_out[k] for k in arrays_out if k.startswith('cls_level_')],
            axis=0)
        arrays_out['cls_mean_score_distribution'] = (
            mean_scores / max(float(mean_scores.sum()), 1.0e-12))
    return rows


def detection_summary(angle, image_name, detections):
    names = detections['class_names']
    scores = np.asarray(detections['scores'], dtype=np.float32)
    counts = Counter(names)
    total = int(len(names))
    small = int(counts.get('small-vehicle', 0))
    rows = []
    for cls_name, count in sorted(counts.items()):
        cls_scores = [s for s, n in zip(scores, names) if n == cls_name]
        rows.append(dict(
            angle=angle,
            image=image_name,
            class_name=cls_name,
            count=int(count),
            mean_score=float(np.mean(cls_scores)) if cls_scores else 0.0))
    summary = dict(
        angle=angle,
        image=image_name,
        detection_total=total,
        small_vehicle_count=small,
        small_vehicle_ratio=float(small / total) if total else 0.0,
        mean_score=float(scores.mean()) if len(scores) else 0.0,
        max_score=float(scores.max()) if len(scores) else 0.0,
        class_histogram=dict(sorted(counts.items())))
    return summary, rows


def main():
    args = parse_args()
    setup_reproducibility(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'features').mkdir(exist_ok=True)
    (out_dir / 'detections').mkdir(exist_ok=True)

    angles = selected_angles(args, args.image_dir)
    support_path, support_warning = resolve_support_path(args.support_feat)
    runner, model, cfg = build_runner_model(args, support_path)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    _, _, det_support_data, name2id, id2name = prepare_support(args, device)
    class_names = list(det_support_data.keys())

    recorder = HookRecorder(args.max_spatial_size)
    hooks = recorder.register(model)
    inv_csv, inv_md = write_module_inventory(
        model, hooks, out_dir, Path(args.result_md_dir))

    loader = build_dataloader(args, angles)
    all_stage_rows = []
    all_cls_rows = []
    det_summary_rows = []
    det_class_rows = []
    per_image_json = {}

    metadata = dict(
        generated_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        config=str(args.config),
        checkpoint=str(args.checkpoint),
        image_dir=str(args.image_dir),
        angles=angles,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        support_type=args.support_type,
        support_shot=args.support_shot,
        support_feat=str(support_path),
        support_warning=support_warning,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        torch_cuda_available=torch.cuda.is_available(),
        device=str(device),
        class_names=class_names,
        module_inventory_csv=str(inv_csv),
        module_inventory_md=str(inv_md),
        hooked_modules=[
            dict(stage=stage, module=name, type=module.__class__.__name__,
                 pre_hook=is_pre)
            for stage, name, module, is_pre in hooks
        ])
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    model.eval()
    processed = 0
    with torch.no_grad():
        for data_info in loader:
            img_path = data_info['data_samples'][0].img_path
            angle = angle_from_name(img_path)
            if angle not in angles:
                continue
            image_name = Path(img_path).stem
            print(f'PROBE angle={angle:03d} image={image_name}')
            recorder.clear()
            data = model.data_preprocessor(data_info, False)
            data['inputs'] = data['inputs'].to(device)
            data_samples = data['data_samples']
            feat_x = model.prompt_extract_feats(data['inputs'])
            results = model.prompt_predict(
                deepcopy(feat_x),
                data['inputs'],
                deepcopy(data_samples),
                val_support_data=det_support_data,
                val_name2id=name2id,
                val_using_aux=False,
                rescale=True,
                support_shot=args.support_shot,
                support_type=args.support_type)
            dets = get_pred_results(
                results, id2name, iou_thr=args.iou_thr, score_thr=args.score_thr)
            det = dets[0]
            per_image_json[image_name] = dict(angle=angle, detections=det)
            with open(out_dir / 'detections' / f'{image_name}.json', 'w') as f:
                json.dump(per_image_json[image_name], f, indent=2)

            det_summary, det_class = detection_summary(angle, image_name, det)
            det_summary_rows.append(det_summary)
            det_class_rows.extend(det_class)

            arrays = {}
            for rec in recorder.records.values():
                for row in rec['stats']:
                    row = dict(row)
                    row['angle'] = angle
                    row['image'] = image_name
                    all_stage_rows.append(row)
                arrays.update(rec['arrays'])
            all_cls_rows.extend(summarize_cls_arrays(
                angle, recorder, class_names, arrays))
            np.savez_compressed(
                out_dir / 'features' / f'{image_name}_features.npz',
                **arrays)
            processed += 1

    recorder.close()
    with open(out_dir / 'stage_stats.csv', 'w', newline='') as f:
        fields = ['angle', 'image', 'stage', 'module', 'tensor_path', 'shape',
                  'mean', 'std', 'min', 'max', 'l2_norm']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_stage_rows:
            row = dict(row)
            row['shape'] = json.dumps(row['shape'])
            writer.writerow(row)
    with open(out_dir / 'cls_summary.csv', 'w', newline='') as f:
        fields = ['angle', 'level', 'module', 'top1_class', 'top1_score',
                  'top2_class', 'top2_score', 'margin', 'top5',
                  'small_vehicle_score', 'entropy']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_cls_rows)
    with open(out_dir / 'detections_summary.csv', 'w', newline='') as f:
        fields = ['angle', 'image', 'detection_total', 'small_vehicle_count',
                  'small_vehicle_ratio', 'mean_score', 'max_score',
                  'class_histogram']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in det_summary_rows:
            row = dict(row)
            row['class_histogram'] = json.dumps(row['class_histogram'],
                                                ensure_ascii=True)
            writer.writerow(row)
    with open(out_dir / 'detections_by_class.csv', 'w', newline='') as f:
        fields = ['angle', 'image', 'class_name', 'count', 'mean_score']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(det_class_rows)
    with open(out_dir / 'detections_all.json', 'w') as f:
        json.dump(per_image_json, f, indent=2)

    metadata['processed_images'] = processed
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'DONE processed={processed} out_dir={out_dir}')


if __name__ == '__main__':
    main()
