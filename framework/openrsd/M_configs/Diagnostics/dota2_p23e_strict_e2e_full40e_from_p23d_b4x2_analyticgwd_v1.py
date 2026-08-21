_base_ = './dota2_p23d_strict_e2e_full5e_from_p22b_b4x2_analyticgwd_v1.py'

# P23E: continue the full-DOTA2 strict-E2E P23D run from epoch_5 to epoch_40.
# Launch with --resume <P23D epoch_5.pth> to restore optimizer/scheduler state.
work_dir = (
    'work_dirs/p23_strict_e2e_dota2_full_20260627/'
    'p23e_full40e_from_p23d_epoch5_b4x2_analyticgwd_v1')

load_from = None
resume = False

max_epochs = 40

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
