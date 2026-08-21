_base_ = './hrrsd_p15_oriented_dino_overfit.py'

work_dir = 'work_dirs/p15_scod_hrrsd_20260623/mini128'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(indices=128))
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        indices=128,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/')))
test_dataloader = val_dataloader

max_epochs = 3
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2))
