_base_ = './A12_flex_rtm_v3_1_self_training_Labelver5.py'

use_declip = False
use_ccl = False

# Fine-tune OpenRSD only on the prepared DOTA2 sliced train split.
# Before training, convert txt annfiles with:
#   python M_Tools/Data1_DOTA2/convert_dota2_txt_to_openrsd_pkl.py

dota2_classes = [
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank', 'swimming-pool',
    'tennis-court',
]

metainfo = dict(classes=dota2_classes, palette=[(220, 20, 60)])
train_metainfo = metainfo

load_from = './results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth'
resume = False
frozen_parameters = ['backbone.stem']

batch_size = 8
num_gpus = 1
max_epochs = 48
base_lr = 2e-5
embed_dims = 256
max_iter_per_epoch = 400
source_prob = [1]

img_scale = (832, 832)
val_img_scale = (1024, 1024)
val_support_classes = dota2_classes
val_dataset_flag = 'Data1_DOTA2'

file_client_args = dict(backend='disk')

model = dict(
    use_declip_support=use_declip,
    support_feat_dict=dict(
        _delete_=True,
        Data1_DOTA2='./data/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
    ),
    declip_support_feat_dict=dict(
        Data1_DOTA2='./data/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DeCLIP_support.pkl',
    ),
    val_support_classes=val_support_classes,
    val_dataset_flag=val_dataset_flag,
    support_type='text',
    with_image_rec_losses=False,
    bbox_head=dict(
        use_ccl_loss=use_ccl,
        ccl_loss_weight=0.05,
        ccl_temperature=0.1,
    ),
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

Data1_DOTA2_SS_Train = dict(
    type='DOTADatasetOnline',
    dataset_flag='Data1_DOTA2',
    data_root='data/DOTA2_1024_500/ss_train',
    metainfo=train_metainfo,
    data_prefix=dict(img_path='images/'),
    ann_file='Step6_Format_labels',
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
        datasets=[Data1_DOTA2_SS_Train]),
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
        data_root='./data/DOTA2_1024_500',
        metainfo=metainfo,
        ann_file='ss_val/annfiles',
        data_prefix=dict(img_path='ss_val/images'),
        img_shape=val_img_scale,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=val_pipeline),
)

test_dataloader = val_dataloader

val_evaluator = dict(type='DETAILDOTAMetric', metric='mAP')
test_evaluator = val_evaluator
