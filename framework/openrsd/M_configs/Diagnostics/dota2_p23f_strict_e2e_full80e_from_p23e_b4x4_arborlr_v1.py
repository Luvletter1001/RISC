_base_ = './dota2_p23e_strict_e2e_full40e_from_p23d_b4x2_analyticgwd_v1.py'

# P23F: full-DOTA2 strict-E2E 80E run from the completed P23E epoch-40 model.
# Use Arbor's best optimizer setting found on GSOVD: lr=2e-4, clip_grad max_norm=0.1.
# Intended launch devices: CUDA_VISIBLE_DEVICES=0,1,8,9.
work_dir = (
    'work_dirs/p23_strict_e2e_dota2_full_20260628/'
    'p23f_full80e_from_p23e_epoch40_b4x4_arborlr_v1')

load_from = (
    '/data1/zcy/OpenRSD/work_dirs/p23_strict_e2e_dota2_full_20260627/'
    'p23e_full40e_from_p23d_epoch5_b4x2_analyticgwd_v1/'
    'best_dota_mAP_epoch_40.pth')
resume = False

max_epochs = 80

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='dota/mAP',
        rule='greater',
        max_keep_ckpts=5,
        save_last=True))

optim_wrapper = dict(
    optimizer=dict(lr=2e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2))

param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[55, 70], gamma=0.1),
]

batch_size = 4
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    dataset=dict(indices=None))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    dataset=dict(indices=None))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=16)
