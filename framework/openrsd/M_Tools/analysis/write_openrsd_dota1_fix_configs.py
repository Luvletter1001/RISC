#!/usr/bin/env python3
"""Write DOTA1 fix configs for F3/F4 follow-up reruns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DOTA1_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter',
]


CUSTOM_IMPORTS_FOR_F4 = [
    'M_AD.engine.runner.meta_remove_runer',
    'M_AD.datasets.transforms.formatting',
    'M_AD.datasets.transforms.loading',
    'M_AD.datasets.transforms.transforms',
    'M_AD.datasets.dota_online_v1',
    'M_AD.datasets.samplers.one_task_sampler',
    'M_AD.models.task_modules.assigners.safe_dynamic_soft_label_assigner',
    'M_AD.models.detectors.Flex_Rtmdet_v3_1_formal',
    'M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1',
    'M_AD.models.roi_heads.CLIP_VP_head_v1',
    'M_AD.models.necks.promopt_cspnext_pafpn',
    'M_AD.evaluation.metrics.detail_dota_metric',
    'M_Tools.analysis.openrsd_cross_view_consistency',
]


def _py_list(items: list[str]) -> str:
    return '[' + ', '.join(repr(x) for x in items) + ']'


def write_config(path: Path,
                 repo_root: Path,
                 train_root: Path,
                 val_root: Path,
                 work_dir: Path,
                 checkpoint: Path,
                 max_iter_per_epoch: int,
                 batch_size: int,
                 model_type: str = 'OpenRTMDet') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_imports = ''
    extra_model = ''
    if model_type == 'OpenRTMDetCrossViewConsistency':
        extra_imports = (
            '\ncustom_imports = dict(\n'
            f'    imports={_py_list(CUSTOM_IMPORTS_FOR_F4)},\n'
            '    allow_failed_imports=False,\n'
            ')\n'
        )
        extra_model = (
            "    type='OpenRTMDetCrossViewConsistency',\n"
            '    consistency_angles=(30, 60, 90, 120, 150),\n'
            '    lambda_cls=0.1,\n'
            '    lambda_box=0.0,\n'
        )

    content = f'''_base_ = '{repo_root / 'M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_DOTA2only_ss_train.py'}'
{extra_imports}
dota1_classes = {_py_list(DOTA1_CLASSES)}
metainfo = dict(classes=dota1_classes, palette=[(220, 20, 60)])
train_metainfo = metainfo

file_client_args = dict(backend='disk')
embed_dims = 256
img_scale = (832, 832)
val_img_scale = (1024, 1024)

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotationsOnline', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxTypeSafe', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=1.0, angle_range=180),
    dict(
        type='mmdet.Pad',
        size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputsMM'),
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=val_img_scale, keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad',
        size=val_img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

load_from = '{checkpoint}'
resume = False
frozen_parameters = ['backbone.stem']

batch_size = {batch_size}
num_gpus = 1
max_epochs = 1
base_lr = 2e-6
max_iter_per_epoch = {max_iter_per_epoch}
source_prob = [1]
val_support_classes = dota1_classes
val_dataset_flag = 'Data1_DOTA2'

model = dict(
{extra_model}    support_feat_dict=dict(
        _delete_=True,
        Data1_DOTA2='{repo_root / 'data/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'}',
    ),
    val_support_classes=val_support_classes,
    val_dataset_flag=val_dataset_flag,
    pca_meta_pth='{repo_root / 'data/7_25_pca_meta_DINOv2_256.pkl'}',
    neg_support_data='{repo_root / 'data/Neg_supports_v2.pkl'}',
    normalized_class_dict='{repo_root / 'data/normalized_class_dict.pkl'}',
    support_type='text',
    val_using_aux=False,
    with_image_rec_losses=False,
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=max_epochs + 1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=20),
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
)

train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='OneTaskSampler',
        batch_size=batch_size,
        source_prob=source_prob,
        num_gpus=num_gpus,
        max_iter_per_epoch=max_iter_per_epoch),
    dataset=dict(
        type='mmdet.ConcatDataset',
        datasets=[
            dict(
                type='DOTADatasetOnline',
                dataset_flag='Data1_DOTA2',
                data_root='{train_root}',
                metainfo=train_metainfo,
                data_prefix=dict(img_path='images/'),
                ann_file='Step6_Format_labels',
                embed_dims=embed_dims,
                img_shape=img_scale,
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=train_pipeline,
            )
        ],
    ),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root='{val_root}',
        metainfo=metainfo,
        ann_file='annfiles',
        data_prefix=dict(img_path='images'),
        img_shape=val_img_scale,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader
val_evaluator = dict(type='DETAILDOTAMetric', metric='mAP')
test_evaluator = val_evaluator
work_dir = '{work_dir}'
'''
    path.write_text(content, encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, default=Path('/data1/zcy/OpenRSD'))
    parser.add_argument('--train-root', type=Path, default=Path('/data1/zcy/OpenRSD/data/DOTA1_1024_500/ss_train'))
    parser.add_argument('--val-root', type=Path, default=Path('/data1/zcy/OpenRSD/work_dirs/openrsd_ovd_rotation_20260508/subsets/dota1/angle_000_n32'))
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('/data1/zcy/OpenRSD/results/MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train_textcls00/epoch_12.pth'))
    parser.add_argument('--max-iters', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = args.work_dir / 'configs'
    f3_config = config_dir / 'f3_dota1_baseline_short_ft.py'
    f4_config = config_dir / 'f4_dota1_consistency_cls.py'
    write_config(
        f3_config, args.repo_root, args.train_root, args.val_root,
        args.work_dir / 'F3_baseline_short_ft', args.checkpoint,
        args.max_iters, args.batch_size, 'OpenRTMDet')
    write_config(
        f4_config, args.repo_root, args.train_root, args.val_root,
        args.work_dir / 'F4_consistency_cls', args.checkpoint,
        args.max_iters, args.batch_size, 'OpenRTMDetCrossViewConsistency')
    manifest = {
        'f3_config': str(f3_config),
        'f4_config': str(f4_config),
        'train_root': str(args.train_root),
        'val_root': str(args.val_root),
        'checkpoint': str(args.checkpoint),
        'max_iters': args.max_iters,
        'batch_size': args.batch_size,
    }
    (args.work_dir / 'config_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
