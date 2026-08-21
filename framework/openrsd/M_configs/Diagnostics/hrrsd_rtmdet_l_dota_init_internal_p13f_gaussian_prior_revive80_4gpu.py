_base_ = './hrrsd_rtmdet_l_dota_init_internal_p13f_gaussian_prior_e2e_4gpu.py'

max_epochs = 80
val_interval = 2
ckpt_interval = 4
base_lr = 0.004 / 16

work_dir = (
    'work_dirs/e2e_second_review80_hrrsd_20260623/'
    'p13f_gaussian_prior')

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

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=ckpt_interval,
        save_last=True,
        max_keep_ckpts=8,
        save_best='dota/mAP',
        rule='greater'))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_interval)

train_dataloader = dict(batch_size=2)

model = dict(
    bbox_head=dict(
        score_thr=0.0,
        max_per_img=260))
