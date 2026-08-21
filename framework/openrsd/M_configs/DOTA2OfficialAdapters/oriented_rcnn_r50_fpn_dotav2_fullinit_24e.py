_base_ = './oriented_rcnn_r50_fpn_dotav2_fullinit.py'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

work_dir = 'work_dirs/dotav2_official_adapters/oriented_rcnn_r50_fpn_fullinit_24e'
