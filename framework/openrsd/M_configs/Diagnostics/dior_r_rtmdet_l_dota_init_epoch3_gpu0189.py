_base_ = '../G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M10_RTMDet_L.py'

data_root = '/data1/zcy/datasets/DIOR_R_dota/'
work_dir = 'work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/train_epoch3'

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
        ann_file='train_val/labelTxt/',
        data_prefix=dict(img_path='train_val/images/')))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        ann_file='test/labelTxt/',
        data_prefix=dict(img_path='test/images/')))

test_dataloader = val_dataloader

model = dict(backbone=dict(init_cfg=None))
