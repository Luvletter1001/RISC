#!/usr/bin/env python3
"""Track one physical court box across rotations and plot class logits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import site
import sys
from collections import Counter, OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.openrsd_env import preload_installed_mmengine

preload_installed_mmengine()

import cv2
import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import DATA_SAMPLERS, DATASETS, FUNCTIONS, init_default_scope
from mmengine.runner.checkpoint import load_checkpoint
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules
from torch.utils.data import DataLoader


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / 'vis' / 'P0148__1024__651___0' / 'dataset')
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / 'workdir_vis' / 'court_box_logit_scan' /
    'P0148__1024__651___0')

DEFAULT_MODEL_SPECS = OrderedDict([
    ('base', dict(
        config=(
            PROJECT_ROOT / 'M_configs' / 'Step2_A10_Large_Pretrain_Stage3' /
            'A10_flex_rtm_v3_1_formal.py'),
        checkpoint=(
            PROJECT_ROOT / 'results' / 'MMR_AD_A10_flex_rtm_v3_1_formal' /
            'epoch_24_weights_only.pth'),
        support_type='text',
    )),
    ('finetune', dict(
        config=(
            PROJECT_ROOT / 'results' /
            'MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train' /
            'A12_flex_rtm_v3_1_DOTA2only_ss_train.py'),
        checkpoint=(
            PROJECT_ROOT / 'results' /
            'MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train' /
            'epoch_12.pth'),
        support_type='text',
    )),
    ('notext_cls', dict(
        config=(
            PROJECT_ROOT / 'results' /
            'MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train_textcls00' /
            'A12_flex_rtm_v3_1_DOTA2only_ss_train.py'),
        checkpoint=(
            PROJECT_ROOT / 'results' /
            'MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train_textcls00' /
            'epoch_12.pth'),
        support_type='text',
    )),
])

ANGLE_PATTERN = re.compile(r'_rot(\d{3})(?:\.[^.]+)?$')
CANONICAL_TRACE_CLASSES = [
    'small-vehicle',
    'tennis-court',
    'basketball-court',
    'baseball-diamond',
]
CLASS_ALIASES = {
    'small-vehicle': {'small-vehicle', 'small_vehicle', 'smallvehicle'},
    'tennis-court': {'tennis-court', 'tennis_court', 'tenniscourt'},
    'basketball-court': {
        'basketball-court', 'basketball_court', 'basketballcourt'
    },
    'baseball-diamond': {
        'baseball-diamond', 'baseball_diamond', 'baseballdiamond',
        'baseball-field', 'baseball_field', 'baseballfield'
    },
}
MODEL_COLORS = OrderedDict([
    ('base', '#f44b8b'),
    ('finetune', '#6c55ff'),
    ('notext_cls', '#28b86a'),
])
TRACE_COLORS = OrderedDict([
    ('small-vehicle', '#d7263d'),
    ('tennis-court', '#2a9d8f'),
    ('basketball-court', '#1d4ed8'),
    ('baseball-diamond', '#b7791f'),
])
PATCHED = False
IMAGE_CENTER = np.array([511.5, 511.5], dtype=np.float32)


def strip_user_site_paths() -> None:
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        user_sites = {Path(user_site).resolve()}
    else:
        user_sites = {Path(path).resolve() for path in user_site}

    cleaned = []
    cwd = Path.cwd().resolve()
    for path_item in sys.path:
        abs_path = cwd if path_item == '' else Path(path_item).resolve()
        if abs_path in user_sites:
            continue
        cleaned.append(path_item)
    sys.path = cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Track one court box over 72 rotations and export logits.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset-root',
        default=str(DEFAULT_DATASET_ROOT),
        help='Dataset root containing images/ and annfiles/.')
    parser.add_argument(
        '--img-dir',
        default='images',
        help='Image subdirectory under dataset-root.')
    parser.add_argument(
        '--ann-dir',
        default='annfiles',
        help='Annotation subdirectory under dataset-root.')
    parser.add_argument(
        '--out-dir',
        default=str(DEFAULT_OUT_DIR),
        help='Output directory for dumps, tables, and plots.')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Torch device used for inference.')
    parser.add_argument(
        '--models',
        default=','.join(DEFAULT_MODEL_SPECS.keys()),
        help='Comma-separated model names to run.')
    parser.add_argument(
        '--support-shot',
        type=int,
        default=8,
        help='Fixed support shot used for every angle.')
    parser.add_argument(
        '--support-seed',
        type=int,
        default=2024,
        help='Seed reused before every prompt selection.')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.001,
        help='Low score threshold used inside the model export.')
    parser.add_argument(
        '--nms-iou',
        type=float,
        default=0.1,
        help='NMS IoU threshold used inside the model export.')
    parser.add_argument(
        '--nms-pre',
        type=int,
        default=4000,
        help='Top-k kept before NMS during export.')
    parser.add_argument(
        '--max-per-img',
        type=int,
        default=4000,
        help='Max detections per image kept after NMS during export.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for the export dataloader.')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Num workers for the export dataloader.')
    parser.add_argument(
        '--sampler-seed',
        type=int,
        default=2024,
        help='Sampler seed for the dataloader.')
    parser.add_argument(
        '--angles',
        default='',
        help='Optional comma-separated subset of angles, e.g. 0,65,100.')
    parser.add_argument(
        '--reference-model',
        default='finetune',
        help='Model used to choose the reference court box.')
    parser.add_argument(
        '--reference-angle',
        type=int,
        default=0,
        help='Angle used to choose the reference court box.')
    parser.add_argument(
        '--reference-class',
        default='tennis-court',
        choices=['tennis-court', 'basketball-court', 'baseball-diamond'],
        help='Reference class used to choose the anchor box.')
    parser.add_argument(
        '--reference-rank',
        type=int,
        default=0,
        help='Rank among detections of the reference class at reference-angle.')
    parser.add_argument(
        '--match-min-iou',
        type=float,
        default=0.05,
        help='IoU threshold below which a match is flagged as weak.')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate raw logit dumps even if cached dumps exist.')
    return parser.parse_args()


def normalize_label(label: str) -> str:
    text = str(label).strip().lower().replace('_', '-')
    text = re.sub(r'[^a-z0-9\-]+', '', text)
    return text


def canonicalize_label(label: str) -> str:
    normalized = normalize_label(label)
    compact = normalized.replace('-', '')
    for canonical, aliases in CLASS_ALIASES.items():
        if normalized in aliases or compact in aliases:
            return canonical
    return normalized


def parse_angle(name: str) -> int:
    match = ANGLE_PATTERN.search(name)
    if not match:
        raise ValueError(f'failed to parse rotation angle from: {name}')
    return int(match.group(1))


def parse_angle_subset(text: str) -> list[int] | None:
    if not text.strip():
        return None
    angles = []
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        angles.append(int(chunk))
    return sorted(set(angles))


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    return obj


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Path) -> Any:
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def resolve_runtime_path(path_value: str | Path, *, config_path: Path) -> Path:
    raw_path = Path(path_value)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([
            Path.cwd() / raw_path,
            PROJECT_ROOT / raw_path,
            config_path.parent / raw_path,
        ])

    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered_candidates.append(resolved)
        if candidate.name == 'Step5_3_Prepare_Visual_Text_DINOv2_support.pkl':
            alt = Path(str(candidate).replace('/train/', '/ss_train/', 1)).resolve()
            if alt not in seen:
                seen.add(alt)
                ordered_candidates.append(alt)

    for candidate in ordered_candidates:
        if candidate.exists():
            return candidate

    if raw_path.name:
        matches = sorted((PROJECT_ROOT / 'data').rglob(raw_path.name))
        if len(matches) == 1:
            return matches[0].resolve()

    return ordered_candidates[0] if ordered_candidates else raw_path


def prepare_model_cfg_for_analysis(cfg: Config, *, config_path: Path) -> None:
    for key in ('normalized_class_dict', 'neg_support_data', 'pca_meta_pth'):
        path_value = cfg.model.get(key)
        if path_value:
            cfg.model[key] = str(
                resolve_runtime_path(path_value, config_path=config_path))

    support_feat_dict = cfg.model.get('support_feat_dict')
    val_dataset_flag = cfg.model.get('val_dataset_flag')
    if support_feat_dict and val_dataset_flag in support_feat_dict:
        # The analysis path only uses validation prompts, so loading one
        # resolved support source is sufficient and avoids stale training
        # entries from older configs.
        cfg.model.support_feat_dict = {
            val_dataset_flag: str(
                resolve_runtime_path(
                    support_feat_dict[val_dataset_flag],
                    config_path=config_path))
        }


def patch_head_for_logits() -> None:
    global PATCHED
    if PATCHED:
        return

    import M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1 as flex_head_mod

    cls = flex_head_mod.OpenRotatedRTMDetSepBNHead

    def _predict_by_feat_single_with_logits(
            self,
            cls_score_list,
            bbox_pred_list,
            angle_pred_list,
            score_factor_list,
            mlvl_priors,
            img_meta,
            cfg,
            rescale=False,
            with_nms=True):
        if score_factor_list[0] is None:
            with_score_factors = False
        else:
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = flex_head_mod.copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_logits = []
        mlvl_prob_vectors = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None

        for cls_score, bbox_pred, angle_pred, score_factor, priors in zip(
                cls_score_list, bbox_pred_list, angle_pred_list,
                score_factor_list, mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1)
                score_factor = score_factor.sigmoid()

            cls_out_channels = cls_score.shape[0]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, cls_out_channels)
            raw_logits = cls_score
            if self.use_sigmoid_cls:
                score_matrix = cls_score.sigmoid()
            else:
                score_matrix = cls_score.softmax(-1)[:, :-1]

            results = flex_head_mod.filter_scores_and_topk(
                score_matrix,
                cfg.get('score_thr', 0),
                nms_pre,
                dict(
                    bbox_pred=bbox_pred,
                    angle_pred=angle_pred,
                    priors=priors,
                    logits=raw_logits,
                    prob_vectors=score_matrix,
                ))
            scores, labels, keep_idxs, filtered = results

            bbox_pred = filtered['bbox_pred']
            angle_pred = filtered['angle_pred']
            priors = filtered['priors']
            raw_logits = filtered['logits']
            prob_vectors = filtered['prob_vectors']

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            if with_score_factors:
                score_factor = score_factor[keep_idxs]
                mlvl_score_factors.append(score_factor)

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_logits.append(raw_logits)
            mlvl_prob_vectors.append(prob_vectors)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = flex_head_mod.cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = flex_head_mod.InstanceData()
        results.bboxes = flex_head_mod.RotatedBoxes(bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        results.logits = torch.cat(mlvl_logits)
        results.prob_vectors = torch.cat(mlvl_prob_vectors)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    cls._predict_by_feat_single = _predict_by_feat_single_with_logits
    PATCHED = True


def build_test_dataloader(
        *,
        dataset_root: Path,
        img_dir: str,
        ann_dir: str,
        class_names: list[str],
        batch_size: int,
        num_workers: int,
        sampler_seed: int) -> DataLoader:
    dataloader_cfg = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='DOTADatasetOnline',
            data_root=str(dataset_root),
            ann_file=ann_dir,
            data_prefix=dict(img_path=img_dir),
            img_shape=(1024, 1024),
            metainfo=dict(classes=class_names, palette=[(220, 20, 60)]),
            filter_cfg=dict(filter_empty_gt=False),
            test_mode=True,
            pipeline=[
                dict(
                    type='mmdet.LoadImageFromFile',
                    file_client_args=dict(backend='disk')),
                dict(type='LoadAnnotationsOnline', with_bbox=True, box_type='qbox'),
                dict(
                    type='ConvertBoxTypeSafe',
                    box_type_mapping=dict(gt_bboxes='rbox')),
                dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
                dict(
                    type='mmdet.Pad',
                    size=(1024, 1024),
                    pad_val=dict(img=(114, 114, 114))),
                dict(
                    type='PackDetInputsMM',
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                    )),
            ]))

    dataset_cfg = deepcopy(dataloader_cfg['dataset'])
    dataset = DATASETS.build(dataset_cfg)
    sampler_cfg = deepcopy(dataloader_cfg['sampler'])
    sampler = DATA_SAMPLERS.build(
        sampler_cfg,
        default_args=dict(dataset=dataset, seed=sampler_seed))

    collate_fn_cfg = dict(type='pseudo_collate')
    collate_fn_type = collate_fn_cfg.pop('type')
    collate_fn_impl = FUNCTIONS.get(collate_fn_type)
    collate_fn = lambda batch: collate_fn_impl(batch, **collate_fn_cfg)

    return DataLoader(
        dataset,
        batch_size=dataloader_cfg['batch_size'],
        sampler=sampler,
        num_workers=dataloader_cfg['num_workers'],
        collate_fn=collate_fn,
        persistent_workers=dataloader_cfg['persistent_workers'],
        drop_last=dataloader_cfg['drop_last'])


def ordered_class_names(name2id: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(name2id.items(), key=lambda item: item[1])]


def resolve_trace_class_indices(class_names: list[str]) -> dict[str, int]:
    out = {}
    normalized = [canonicalize_label(name) for name in class_names]
    for target in CANONICAL_TRACE_CLASSES:
        if target not in normalized:
            raise KeyError(
                f'class "{target}" not found in model classes: {class_names}')
        out[target] = normalized.index(target)
    return out


def build_model(
        *,
        config_path: Path,
        checkpoint_path: Path,
        device: str,
        score_thr: float,
        nms_iou: float,
        nms_pre: int,
        max_per_img: int):
    cfg = Config.fromfile(str(config_path))
    prepare_model_cfg_for_analysis(cfg, config_path=config_path)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    cfg.model.setdefault('test_cfg', dict())
    cfg.model.test_cfg['score_thr'] = score_thr
    cfg.model.test_cfg['nms_pre'] = nms_pre
    cfg.model.test_cfg['max_per_img'] = max_per_img
    cfg.model.test_cfg['min_bbox_size'] = 0
    cfg.model.test_cfg['nms'] = dict(type='nms_rotated', iou_threshold=nms_iou)

    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    load_checkpoint(model, str(checkpoint_path), map_location='cpu')
    model.to(device)
    model.eval()
    return cfg, model


def maybe_filter_angles(
        data_samples: list[dict[str, Any]],
        keep_angles: set[int] | None) -> bool:
    if keep_angles is None:
        return False
    for sample in data_samples:
        angle = parse_angle(Path(sample['img_path']).stem)
        if angle in keep_angles:
            return False
    return True


def tensor_to_dict(pred_instances, class_names: list[str]) -> dict[str, Any]:
    bboxes = pred_instances.bboxes
    if hasattr(bboxes, 'tensor'):
        bboxes = bboxes.tensor
    return dict(
        bboxes=to_numpy(bboxes).astype(np.float32),
        scores=to_numpy(pred_instances.scores).astype(np.float32),
        labels=to_numpy(pred_instances.labels).astype(np.int64),
        logits=to_numpy(pred_instances.logits).astype(np.float32),
        prob_vectors=to_numpy(pred_instances.prob_vectors).astype(np.float32),
        class_names=list(class_names),
    )


def collect_model_dump(
        *,
        model_name: str,
        config_path: Path,
        checkpoint_path: Path,
        support_type: str,
        args: argparse.Namespace,
        keep_angles: set[int] | None,
        dump_path: Path) -> dict[int, dict[str, Any]]:
    if dump_path.exists() and not args.force:
        cached = load_pickle(dump_path)
        return {int(k): v for k, v in cached['samples_by_angle'].items()}

    cfg, model = build_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=args.device,
        score_thr=args.score_thr,
        nms_iou=args.nms_iou,
        nms_pre=args.nms_pre,
        max_per_img=args.max_per_img,
    )

    class_names = ordered_class_names(model.val_name2id)
    dataloader = build_test_dataloader(
        dataset_root=Path(args.dataset_root),
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        class_names=class_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler_seed=args.sampler_seed)

    samples_by_angle: dict[int, dict[str, Any]] = {}
    total = len(dataloader)

    with torch.no_grad():
        for batch_idx, data_info in enumerate(dataloader, start=1):
            raw_batch_samples = []
            if isinstance(data_info, dict):
                batch_samples = data_info.get('data_samples', [])
                for sample in batch_samples:
                    raw_batch_samples.append(dict(
                        img_path=sample.img_path,
                        img_id=sample.img_id))
            else:
                for item in data_info:
                    raw_batch_samples.append(dict(
                        img_path=item['data_samples'].img_path,
                        img_id=item['data_samples'].img_id))
            if maybe_filter_angles(raw_batch_samples, keep_angles):
                continue

            data = model.data_preprocessor(data_info, False)
            data_samples = data['data_samples']
            feat_x = model.prompt_extract_feats(data['inputs'])

            np.random.seed(args.support_seed)
            prompt_results = model.prompt_predict(
                deepcopy(feat_x),
                data['inputs'],
                deepcopy(data_samples),
                rescale=False,
                support_type=support_type,
                support_shot=args.support_shot,
                val_support_data=model.val_support_data,
                val_name2id=model.val_name2id,
                val_using_aux=False,
            )

            for sample in prompt_results:
                angle = parse_angle(Path(sample.img_path).stem)
                if keep_angles is not None and angle not in keep_angles:
                    continue
                pred_instances = sample.pred_instances
                sample_dict = dict(
                    model_name=model_name,
                    angle=angle,
                    img_id=sample.img_id,
                    img_path=sample.img_path,
                    pred_instances=tensor_to_dict(pred_instances, class_names),
                )
                samples_by_angle[angle] = sample_dict

            print(
                f'[{model_name}] {batch_idx:03d}/{total:03d} '
                f'collected={len(samples_by_angle)}',
                flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    payload = dict(
        model_name=model_name,
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        support_type=support_type,
        support_shot=args.support_shot,
        support_seed=args.support_seed,
        score_thr=args.score_thr,
        nms_iou=args.nms_iou,
        nms_pre=args.nms_pre,
        max_per_img=args.max_per_img,
        samples_by_angle=samples_by_angle,
    )
    save_pickle(payload, dump_path)
    return samples_by_angle


def wrap_le90(angle_rad: np.ndarray) -> np.ndarray:
    return ((angle_rad + math.pi / 2) % math.pi) - math.pi / 2


def rotate_points(points: np.ndarray, angle_deg: float, center: np.ndarray) -> np.ndarray:
    rad = math.radians(angle_deg)
    cos_v = math.cos(rad)
    sin_v = math.sin(rad)
    centered = points - center[None, :]
    x = centered[:, 0] * cos_v - centered[:, 1] * sin_v
    y = centered[:, 0] * sin_v + centered[:, 1] * cos_v
    return np.stack([x, y], axis=1) + center[None, :]


def inverse_rotate_boxes(boxes: np.ndarray, angle_deg: float) -> np.ndarray:
    out = boxes.copy().astype(np.float32)
    out[:, :2] = rotate_points(out[:, :2], -angle_deg, IMAGE_CENTER)
    out[:, 4] = wrap_le90(out[:, 4] - math.radians(angle_deg))
    return out


def rbox_to_poly(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h, angle = [float(v) for v in box]
    dx = w / 2.0
    dy = h / 2.0
    corners = np.array([
        [-dx, -dy],
        [dx, -dy],
        [dx, dy],
        [-dx, dy],
    ], dtype=np.float32)
    cos_v = math.cos(angle)
    sin_v = math.sin(angle)
    rot = np.array([[cos_v, -sin_v], [sin_v, cos_v]], dtype=np.float32)
    return corners @ rot.T + np.array([cx, cy], dtype=np.float32)[None, :]


def draw_reference_box(image_path: Path, box: np.ndarray, out_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    poly = np.round(rbox_to_poly(box)).astype(np.int32)
    cv2.polylines(image, [poly], True, (30, 200, 255), 3, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def select_reference_box(
        *,
        sample: dict[str, Any],
        target_class: str,
        rank: int) -> dict[str, Any]:
    pred = sample['pred_instances']
    class_names = pred['class_names']
    canonical_classes = [canonicalize_label(name) for name in class_names]
    target_index = canonical_classes.index(target_class)

    labels = pred['labels']
    scores = pred['scores']
    logits = pred['logits']
    boxes = pred['bboxes']

    candidate_indices = np.where(labels == target_index)[0]
    if len(candidate_indices) == 0:
        candidate_indices = np.argsort(logits[:, target_index])[::-1]
    else:
        candidate_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

    if rank >= len(candidate_indices):
        raise IndexError(
            f'reference rank {rank} out of range, '
            f'available candidates={len(candidate_indices)}')

    selected = int(candidate_indices[rank])
    return dict(
        selected_index=selected,
        target_index=target_index,
        canonical_class=target_class,
        class_name=class_names[target_index],
        score=float(scores[selected]),
        label_index=int(labels[selected]),
        label_name=class_names[int(labels[selected])],
        box=boxes[selected].astype(np.float32),
        logits=logits[selected].astype(np.float32),
    )


def analyze_trace(
        *,
        samples_by_model: OrderedDict[str, dict[int, dict[str, Any]]],
        reference_box: np.ndarray,
        keep_angles: list[int]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    trace_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}

    for model_name, by_angle in samples_by_model.items():
        first_sample = by_angle[keep_angles[0]]
        trace_indices = resolve_trace_class_indices(
            first_sample['pred_instances']['class_names'])
        matched_labels = []
        valid_ious = []
        weak_match_count = 0

        for angle in keep_angles:
            sample = by_angle[angle]
            pred = sample['pred_instances']
            boxes = pred['bboxes']
            scores = pred['scores']
            labels = pred['labels']
            logits = pred['logits']
            class_names = pred['class_names']

            row = dict(
                model=model_name,
                angle=angle,
                img_id=sample['img_id'],
                img_path=sample['img_path'],
                matched=False,
                matched_iou=float('nan'),
                matched_score=float('nan'),
                matched_label='',
                matched_label_index=-1,
                matched_box_cx=float('nan'),
                matched_box_cy=float('nan'),
                matched_box_w=float('nan'),
                matched_box_h=float('nan'),
                matched_box_angle=float('nan'),
                weak_match=False,
            )
            for trace_class in CANONICAL_TRACE_CLASSES:
                row[f'{trace_class}_logit'] = float('nan')
                row[f'{trace_class}_prob'] = float('nan')

            if len(boxes) == 0:
                trace_rows.append(row)
                continue

            canonical_boxes = inverse_rotate_boxes(boxes, angle)
            ious = box_iou_rotated(
                torch.tensor(canonical_boxes, dtype=torch.float32),
                torch.tensor(reference_box[None, :], dtype=torch.float32),
            ).squeeze(1).cpu().numpy()
            best_idx = int(np.argmax(ious))

            row['matched'] = True
            row['matched_iou'] = float(ious[best_idx])
            row['matched_score'] = float(scores[best_idx])
            row['matched_label_index'] = int(labels[best_idx])
            row['matched_label'] = class_names[int(labels[best_idx])]
            row['matched_box_cx'] = float(boxes[best_idx, 0])
            row['matched_box_cy'] = float(boxes[best_idx, 1])
            row['matched_box_w'] = float(boxes[best_idx, 2])
            row['matched_box_h'] = float(boxes[best_idx, 3])
            row['matched_box_angle'] = float(boxes[best_idx, 4])
            row['weak_match'] = bool(row['matched_iou'] < args.match_min_iou)
            if row['weak_match']:
                weak_match_count += 1

            prob_vector = 1.0 / (1.0 + np.exp(-logits[best_idx]))
            for trace_class, class_idx in trace_indices.items():
                row[f'{trace_class}_logit'] = float(logits[best_idx, class_idx])
                row[f'{trace_class}_prob'] = float(prob_vector[class_idx])

            matched_labels.append(canonicalize_label(row['matched_label']))
            valid_ious.append(row['matched_iou'])
            trace_rows.append(row)

        summaries[model_name] = dict(
            weak_match_count=weak_match_count,
            mean_iou=float(np.mean(valid_ious)) if valid_ious else float('nan'),
            min_iou=float(np.min(valid_ious)) if valid_ious else float('nan'),
            matched_label_counts=dict(Counter(matched_labels)),
        )

    return trace_rows, summaries


def build_wide_rows(trace_rows: list[dict[str, Any]], model_names: list[str]) -> list[dict[str, Any]]:
    rows_by_angle: dict[int, dict[str, Any]] = OrderedDict()
    for angle in sorted({row['angle'] for row in trace_rows}):
        rows_by_angle[angle] = dict(angle=angle)

    for row in trace_rows:
        angle = row['angle']
        prefix = row['model']
        target = rows_by_angle[angle]
        for key, value in row.items():
            if key in {'model', 'angle'}:
                continue
            target[f'{prefix}_{key}'] = value

    return [rows_by_angle[angle] for angle in rows_by_angle]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_trace_curves(
        trace_rows: list[dict[str, Any]],
        out_path: Path,
        model_names: list[str]) -> None:
    strip_user_site_paths()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=len(model_names),
        ncols=1,
        figsize=(14, 4 * len(model_names)),
        sharex=True)
    if len(model_names) == 1:
        axes = [axes]

    for axis, model_name in zip(axes, model_names):
        model_rows = [row for row in trace_rows if row['model'] == model_name]
        model_rows = sorted(model_rows, key=lambda row: row['angle'])
        angles = [row['angle'] for row in model_rows]
        for trace_class, color in TRACE_COLORS.items():
            values = [row[f'{trace_class}_logit'] for row in model_rows]
            axis.plot(
                angles,
                values,
                color=color,
                linewidth=2.2,
                marker='o',
                markersize=3.2,
                label=trace_class)
        weak_angles = [row['angle'] for row in model_rows if row['weak_match']]
        for weak_angle in weak_angles:
            axis.axvline(weak_angle, color='#999999', linewidth=0.8, alpha=0.35)
        axis.set_title(model_name)
        axis.set_ylabel('logit')
        axis.grid(True, alpha=0.25, linewidth=0.8)
        axis.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('rotation angle (deg)')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_match_iou(
        trace_rows: list[dict[str, Any]],
        out_path: Path,
        model_names: list[str]) -> None:
    strip_user_site_paths()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4.2))
    for model_name in model_names:
        model_rows = [row for row in trace_rows if row['model'] == model_name]
        model_rows = sorted(model_rows, key=lambda row: row['angle'])
        ax.plot(
            [row['angle'] for row in model_rows],
            [row['matched_iou'] for row in model_rows],
            color=MODEL_COLORS[model_name],
            linewidth=2.0,
            marker='o',
            markersize=3.2,
            label=model_name)
    ax.set_xlabel('rotation angle (deg)')
    ax.set_ylabel('IoU vs reference box')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(loc='lower left')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def summarize_trace(
        *,
        trace_rows: list[dict[str, Any]],
        model_names: list[str],
        reference_payload: dict[str, Any],
        summaries: dict[str, dict[str, Any]],
        out_path: Path) -> None:
    lines = []
    lines.append('# Court Box Logit Trace Summary')
    lines.append('')
    lines.append(f'- Reference model: `{reference_payload["model_name"]}`')
    lines.append(f'- Reference angle: `{reference_payload["reference_angle"]}`')
    lines.append(f'- Reference class: `{reference_payload["canonical_class"]}`')
    lines.append(f'- Reference score: `{reference_payload["score"]:.4f}`')
    lines.append(
        f'- Reference box: `{[round(float(v), 4) for v in reference_payload["box"]]}`')
    lines.append('')

    for model_name in model_names:
        model_rows = [row for row in trace_rows if row['model'] == model_name]
        lines.append(f'## {model_name}')
        info = summaries[model_name]
        lines.append(f'- Mean IoU to reference: `{info["mean_iou"]:.4f}`')
        lines.append(f'- Min IoU to reference: `{info["min_iou"]:.4f}`')
        lines.append(f'- Weak-match angles: `{info["weak_match_count"]}`')
        lines.append(
            f'- Matched labels: `{json.dumps(info["matched_label_counts"], ensure_ascii=False)}`')
        for trace_class in CANONICAL_TRACE_CLASSES:
            values = np.array(
                [row[f'{trace_class}_logit'] for row in model_rows], dtype=np.float32)
            lines.append(
                f'- {trace_class} logit mean/std: '
                f'`{float(np.nanmean(values)):.4f} / {float(np.nanstd(values)):.4f}`')
        small_values = np.array(
            [row['small-vehicle_logit'] for row in model_rows], dtype=np.float32)
        tennis_values = np.array(
            [row['tennis-court_logit'] for row in model_rows], dtype=np.float32)
        if np.isfinite(small_values).sum() >= 2 and np.isfinite(tennis_values).sum() >= 2:
            corr = float(np.corrcoef(small_values, tennis_values)[0, 1])
            lines.append(f'- Corr(small-vehicle, tennis-court): `{corr:.4f}`')
        lines.append('')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines), encoding='utf-8')


def save_run_config(args: argparse.Namespace, model_names: list[str], out_path: Path) -> None:
    payload = dict(
        args=to_builtin(vars(args)),
        model_names=model_names,
        model_specs=to_builtin({
            name: DEFAULT_MODEL_SPECS[name] for name in model_names
        }))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def main() -> None:
    global args
    args = parse_args()
    keep_angles = parse_angle_subset(args.angles)
    keep_angle_set = set(keep_angles) if keep_angles is not None else None

    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)
    patch_head_for_logits()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = [name.strip() for name in args.models.split(',') if name.strip()]
    for model_name in model_names:
        if model_name not in DEFAULT_MODEL_SPECS:
            raise KeyError(f'unknown model name: {model_name}')
    if args.reference_model not in model_names:
        if len(model_names) == 1:
            args.reference_model = model_names[0]
        else:
            raise KeyError(
                f'reference model "{args.reference_model}" is not in the '
                f'requested models: {model_names}')

    samples_by_model: OrderedDict[str, dict[int, dict[str, Any]]] = OrderedDict()
    for model_name in model_names:
        spec = DEFAULT_MODEL_SPECS[model_name]
        dump_path = out_dir / f'{model_name}_logit_dump.pkl'
        samples_by_model[model_name] = collect_model_dump(
            model_name=model_name,
            config_path=Path(spec['config']),
            checkpoint_path=Path(spec['checkpoint']),
            support_type=str(spec['support_type']),
            args=args,
            keep_angles=keep_angle_set,
            dump_path=dump_path)

    available_angles = sorted(set.intersection(
        *(set(by_angle.keys()) for by_angle in samples_by_model.values())))
    if not available_angles:
        raise RuntimeError('no common angles found across model dumps')
    if keep_angles is None:
        keep_angles = available_angles

    reference_sample = samples_by_model[args.reference_model][args.reference_angle]
    reference_payload = select_reference_box(
        sample=reference_sample,
        target_class=args.reference_class,
        rank=args.reference_rank)
    reference_payload['model_name'] = args.reference_model
    reference_payload['reference_angle'] = args.reference_angle

    reference_img_path = Path(reference_sample['img_path'])
    draw_reference_box(
        reference_img_path,
        reference_payload['box'],
        out_dir / 'reference_box_overlay.png')
    (out_dir / 'reference_box.json').write_text(
        json.dumps(to_builtin(reference_payload), indent=2, ensure_ascii=False),
        encoding='utf-8')

    trace_rows, summaries = analyze_trace(
        samples_by_model=samples_by_model,
        reference_box=reference_payload['box'],
        keep_angles=keep_angles)
    trace_rows = sorted(trace_rows, key=lambda row: (row['model'], row['angle']))
    wide_rows = build_wide_rows(trace_rows, model_names)

    write_csv(out_dir / 'court_box_logit_trace_long.csv', trace_rows)
    write_csv(out_dir / 'court_box_logit_trace_wide.csv', wide_rows)
    plot_trace_curves(trace_rows, out_dir / 'court_box_logit_curve.png', model_names)
    plot_match_iou(trace_rows, out_dir / 'court_box_match_iou.png', model_names)
    summarize_trace(
        trace_rows=trace_rows,
        model_names=model_names,
        reference_payload=reference_payload,
        summaries=summaries,
        out_path=out_dir / 'court_box_logit_summary.md')
    save_run_config(args, model_names, out_dir / 'run_config.json')

    print(f'Saved outputs to: {out_dir}')


if __name__ == '__main__':
    main()
