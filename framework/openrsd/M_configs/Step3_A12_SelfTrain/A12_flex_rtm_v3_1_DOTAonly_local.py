_base_ = './A12_flex_rtm_v3_1_self_training_Labelver5.py'

# Local DOTA fine-tuning config.
# Train labels must be converted once with:
#   PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python \
#     M_Tools/Data1_DOTA_Local/convert_dota_txt_to_openrsd_pkl.py

dota_classes = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter',
]

metainfo = dict(classes=dota_classes, palette=[(220, 20, 60)])
train_metainfo = metainfo

load_from = './results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth'
resume = False
frozen_parameters = ['backbone.stem']

batch_size = 4
num_gpus = 4
max_epochs = 12
base_lr = 2e-5
embed_dims = 256
max_iter_per_epoch = 800
source_prob = [1]

img_scale = (832, 832)
val_img_scale = (1024, 1024)
val_support_classes = dota_classes
val_dataset_flag = 'Data1_DOTA2'

file_client_args = dict(backend='disk')

model = dict(
    support_feat_dict=dict(
        _delete_=True,
        Data1_DOTA2='./data/DOTA_800_600/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
    ),
    val_support_classes=val_support_classes,
    val_dataset_flag=val_dataset_flag,
    support_type='text',
    with_image_rec_losses=False,
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=4,
        end=max_epochs,
        T_max=max_epochs - 4,
        by_epoch=True,
        convert_to_iter_based=True),
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
)

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotationsOnline', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxTypeSafe', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=0.5, angle_range=180),
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

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=val_img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size=val_img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

Data1_DOTA2_Local = dict(
    type='DOTADatasetOnline',
    dataset_flag='Data1_DOTA2',
    data_root='.',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='/data/zcy/dataset/trainval_ss/images'),
    ann_file='./data/OpenRSD_DOTA_trainval_ss/annfiles',
    embed_dims=embed_dims,
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline,
)

train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='OneTaskSampler',
        batch_size=batch_size,
        source_prob=source_prob,
        num_gpus=num_gpus,
        max_iter_per_epoch=max_iter_per_epoch),
    dataset=dict(
        type='mmdet.ConcatDataset',
        datasets=[Data1_DOTA2_Local]),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root='/data/zcy/dataset/dota15/val',
        metainfo=metainfo,
        ann_file='labelTxt',
        data_prefix=dict(img_path='images'),
        img_shape=val_img_scale,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=val_pipeline),
)

test_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root='/data/zcy/dataset/test_ss',
        metainfo=metainfo,
        ann_file='',
        data_prefix=dict(img_path='images'),
        img_shape=val_img_scale,
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=test_pipeline),
)

val_evaluator = dict(type='DETAILDOTAMetric', metric='mAP')
test_evaluator = dict(
    type='DOTAMetric',
    metric='mAP',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./results/MMR_AD_A12_flex_rtm_v3_1_DOTAonly_local/dota_submit')
