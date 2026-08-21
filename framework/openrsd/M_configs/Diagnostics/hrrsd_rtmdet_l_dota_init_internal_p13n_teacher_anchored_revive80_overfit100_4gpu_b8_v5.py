_base_ = './hrrsd_rtmdet_l_dota_init_internal_p13n_teacher_anchored_posterior_e2e_4gpu.py'

data_root = '/data1/zcy/datasets/HRRSD_800_0/internal_split_20260619/'

max_epochs = 80
val_interval = 5
ckpt_interval = 5
base_lr = 0.004 / 16

work_dir = (
    'work_dirs/e2e_revive80_hrrsd_20260623/'
    'p13n_teacher_anchored_overfit100_4gpu_b8_v5')

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_interval)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        interval=ckpt_interval,
        save_last=True,
        max_keep_ckpts=10,
        save_best='dota/mAP',
        rule='greater'))

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
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(indices=100))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/'),
        indices=100,
        filter_cfg=dict(filter_empty_gt=False)))

test_dataloader = val_dataloader

model = dict(
    bbox_head=dict(
        teacher_predictions_path=(
            'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
            'teacher_p13c_dense_fdd_obj_train_full/predictions.pkl'),
        score_thr=0.0,
        max_per_img=520))

auto_scale_lr = dict(enable=False, base_batch_size=32)
