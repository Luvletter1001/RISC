_base_ = './dota2_p22b_strict_e2e_train1000_curriculum_b16x4_v1.py'

# P23A: continue the strict E2E DOTA2 line from the P22B 1000-image best
# checkpoint, but train on the full sliced DOTA2 ss_train split.
#
# Resource constraint from user: use GPU 8/9 only. Keep b16/GPU because b32/GPU
# OOMed on DOTA2 dense scenes in P22A.
work_dir = (
    'work_dirs/p23_strict_e2e_dota2_full_20260626/'
    'p23a_full5e_from_p22b_b16x2_v1')

load_from = (
    'work_dirs/p22_strict_e2e_dota2_subset_20260626/'
    'p22b_train1000_curriculum_from_p21a_b16x4_v1/'
    'best_dota_mAP_epoch_80.pth')
resume = False

max_epochs = 5

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
    optimizer=dict(lr=2e-5))

param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[3],
        gamma=0.1),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(indices=None))

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(indices=None))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=32)
