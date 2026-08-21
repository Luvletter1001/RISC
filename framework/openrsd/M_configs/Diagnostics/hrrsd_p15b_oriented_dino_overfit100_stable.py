_base_ = './hrrsd_p15b_oriented_dino_overfit.py'

work_dir = 'work_dirs/p15b_scod_hrrsd_20260623/overfit100_stable'

max_epochs = 80
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_last=True,
        max_keep_ckpts=10,
        save_best='dota/mAP',
        rule='greater'))

train_dataloader = dict(dataset=dict(indices=100))
val_dataloader = dict(dataset=dict(indices=100))
test_dataloader = val_dataloader

param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[55, 70], gamma=0.1)
]
