_base_ = (
    '/data1/zcy/OpenRSD/work_dirs/gpu67_controls_20260615/'
    'c2b_noop_lr0_frozenbn_2e/c2b_noop_lr0_frozenbn_2e.py'
)

dota2_classes = [
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank', 'swimming-pool',
    'tennis-court',
]

data_root = '/data1/zcy/datasets/DOTA2_1024_500'
support_root = '/data1/zcy/datasets/OPENRSD_DATA/runtime_meta_risc_20260807'
train_label_dir = 'Step6_Format_labels_risc_20260807'
c2b_checkpoint = (
    '/data1/zcy/OpenRSD/work_dirs/gpu67_controls_20260615/'
    'c2b_noop_lr0_frozenbn_2e/best_dota_mAP_epoch_1.pth'
)

custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'M_AD.engine.runner.meta_remove_runer',
        'M_AD.engine.hooks.freeze_norm_stats_hook',
        'M_AD.datasets.transforms.formatting',
        'M_AD.datasets.transforms.loading',
        'M_AD.datasets.transforms.transforms',
        'M_AD.datasets.dota_online_v1',
        'M_AD.datasets.samplers.one_task_sampler',
        'M_AD.models.task_modules.assigners.safe_dynamic_soft_label_assigner',
        'M_AD.models.detectors.Flex_Rtmdet_v3_1_formal',
        'M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1',
        'M_AD.models.dense_heads.risc_orbit_projection_head',
        'M_AD.models.roi_heads.CLIP_VP_head_v1',
        'M_AD.models.necks.promopt_cspnext_pafpn',
        'M_AD.evaluation.metrics.detail_dota_metric',
    ])

model = dict(
    bbox_head=dict(
        type='RISCOrbitProjectedRTMDetHead',
        use_focus_ovd=True,
        focus_ovd=dict(
            enable=True,
            orientation=dict(
                patch_size=7,
                num_angle_bins=36,
                harmonic_orders=(2, 4, 6),
                confidence_type='peak_entropy',
                min_confidence=0.15,
                detach_orientation=True,
            ),
            adapter=dict(enable=True),
            eqtext=dict(enable=False),
            dual_fusion=dict(enable=False),
        ),
        orbit_projection=dict(
            rank=8,
            init_strength=0.0,
            max_strength=0.10,
            max_delta_norm_ratio=0.05,
            normalize_after_projection=True,
            basis_seed=20260807,
        ),
    ),
    support_type='text',
    use_declip_support=False,
    support_feat_dict=dict(
        _delete_=True,
        Data1_DOTA2=(
            f'{support_root}/'
            'Step5_3_Prepare_Visual_Text_DINOv2_support.pkl')),
    neg_support_data=f'{support_root}/Neg_supports_v2.pkl',
    normalized_class_dict=f'{support_root}/normalized_class_dict.pkl',
    pca_meta_pth=f'{support_root}/7_25_pca_meta_DINOv2_256.pkl',
    val_support_classes=dota2_classes,
    val_dataset_flag='Data1_DOTA2',
    val_using_aux=False,
)

train_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotationsOnline', with_bbox=True, box_type='qbox'),
    dict(
        type='ConvertBoxTypeSafe',
        box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(832, 832), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=1.0, angle_range=180),
    dict(
        type='mmdet.Pad',
        size=(832, 832),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputsMM'),
]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        _delete_=True,
        type='mmdet.ConcatDataset',
        datasets=[
            dict(
                type='DOTADatasetOnline',
                data_root=f'{data_root}/ss_train',
                ann_file=train_label_dir,
                data_prefix=dict(img_path='images/'),
                dataset_flag='Data1_DOTA2',
                embed_dims=256,
                img_shape=(832, 832),
                filter_cfg=dict(filter_empty_gt=True),
                metainfo=dict(
                    classes=dota2_classes,
                    palette=[(220, 20, 60)]),
                pipeline=train_pipeline,
            ),
        ]),
    sampler=dict(
        type='OneTaskSampler',
        batch_size=2,
        num_gpus=2,
        max_iter_per_epoch=400,
        source_prob=[1]),
)

val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        ann_file='ss_val/annfiles',
        data_prefix=dict(img_path='ss_val/images'),
        metainfo=dict(classes=dota2_classes, palette=[(220, 20, 60)]),
        filter_cfg=dict(filter_empty_gt=False),
        img_shape=(1024, 1024)),
)
test_dataloader = val_dataloader

trainable_parameters = ['bbox_head.focus_support_adapter']
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=3e-4, weight_decay=0.0))
param_scheduler = []
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)
num_gpus = 2

load_from = c2b_checkpoint
resume = False
work_dir = (
    '/data1/zcy/OpenRSD/work_dirs/'
    'dotav2_risc_orbit_lowrank_r8_c2b_frozenbn_2e_gpu89_20260807'
)
