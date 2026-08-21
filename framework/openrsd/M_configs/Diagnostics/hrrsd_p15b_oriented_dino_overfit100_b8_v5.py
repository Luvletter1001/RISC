_base_ = './hrrsd_p15b_oriented_dino_overfit100_stable.py'

work_dir = 'work_dirs/p15b_scod_hrrsd_20260623/overfit100_b8_v5'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=5)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_last=True,
        max_keep_ckpts=10,
        save_best='dota/mAP',
        rule='greater'))

train_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
val_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=8)
