_base_ = '../G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M10_RTMDet_L.py'

data_root = '/data1/zcy/datasets/HRRSD_800_0/internal_split_20260619/'

class_name = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])
num_classes = len(class_name)

work_dir = 'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/train_epoch3'

load_from = (
    'weights/rotated_rtmdet_l-3x-dota_ms-2738da34_no_cls_dior_init.pth')
resume = False

max_epochs = 3
val_interval = 4
ckpt_interval = 1

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=ckpt_interval,
        save_last=True,
        max_keep_ckpts=4))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_interval)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/')))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        filter_cfg=dict(filter_empty_gt=False)))

test_dataloader = val_dataloader

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(num_classes=num_classes))
