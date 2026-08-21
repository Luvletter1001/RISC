_base_ = '../G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M9_RTMDet_M.py'

data_root = '/data1/zcy/datasets/DIOR_R_dota/'

max_iters = 200
train_subset = 800
eval_subset = 256

work_dir = 'work_dirs/gs3c_dior_r_network_smoke_20260619/baseline_smoke200'

load_from = None
resume = False

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=max_iters + 1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0, by_epoch=False, begin=0, end=max_iters)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=max_iters,
        save_last=True,
        max_keep_ckpts=2))

custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        data_root=data_root,
        ann_file='train_val/labelTxt/',
        data_prefix=dict(img_path='train_val/images/'),
        indices=train_subset))

val_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        ann_file='test/labelTxt/',
        data_prefix=dict(img_path='test/images/'),
        indices=eval_subset))

test_dataloader = val_dataloader

model = dict(
    backbone=dict(norm_cfg=dict(type='BN'), init_cfg=None),
    neck=dict(norm_cfg=dict(type='BN')),
    bbox_head=dict(norm_cfg=dict(type='BN')))
