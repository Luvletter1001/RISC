_base_ = './xview_rtmdet_l_dota_init_epoch3_gpu0189.py'

work_dir = 'work_dirs/gs3c_xview_rtmdetl_dota_init_20260619/train_epoch3_plus9'

load_from = (
    'work_dirs/gs3c_xview_rtmdetl_dota_init_20260619/'
    'train_epoch3/epoch_3.pth')
resume = False

max_epochs = 9
val_interval = 10
ckpt_interval = 1

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_interval)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=ckpt_interval,
        save_last=True,
        max_keep_ckpts=4))
