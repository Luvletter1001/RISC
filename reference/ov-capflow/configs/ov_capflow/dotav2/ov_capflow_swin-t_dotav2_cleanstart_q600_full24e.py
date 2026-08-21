_base_ = ['./ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped.py']

# The grouped S1 recipe is the current promotable winner. Launch remains gated
# on completion of the registered repeat and Chamfer comparisons.
promoted_s1_winner = '8-S1-G'
physical_gpus = (4, 5, 6, 7)
expected_train_tiles = 47294
expected_raw_val_tiles = 13833
raw_validation_filter_empty_gt = False

data_root = '/data1/zcy/datasets/DOTA2_1024_500/'

# PyTorch 1.12 DDP cannot combine reentrant Swin checkpointing with the
# two-step gradient accumulation used by this four-GPU run.  The combination
# marks the same backbone parameter ready twice after the first update.
# Keeping both switches off preserves the model/loss recipe and spends the
# available A40 memory on activations instead of recomputation.
model = dict(backbone=dict(with_cp=False))
model_wrapper_cfg = dict(static_graph=False)

train_dataloader = dict(
    batch_size=4,
    batch_sampler=dict(
        num_matching_queries=1800,
        max_query_area=50000000,
        update_count_multiple=2,
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            'full24_grouped_train_epoch.json')),
    dataset=dict(
        data_root=data_root,
        ann_file='ss_train/annfiles/',
        data_prefix=dict(img_path='ss_train/images/'),
        filter_cfg=dict(filter_empty_gt=False)))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='ss_val/annfiles/',
        data_prefix=dict(img_path='ss_val/images/'),
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=True))
test_dataloader = val_dataloader

train_cfg = dict(
    _delete_=True,
    type='VariableBatchEpochBasedTrainLoop',
    max_epochs=24,
    val_interval=1,
    dynamic_intervals=[(3, 6)])
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0 / 10, by_epoch=False,
        begin=0, end=500),
    dict(
        type='CosineAnnealingLR', by_epoch=True, begin=0, end=24,
        eta_min=1e-6),
]
optim_wrapper = dict(accumulative_counts=2)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        _delete_=True,
        type='MilestoneCheckpointHook',
        milestones=(1, 6, 12, 18, 24)))

resume = False
work_dir = 'work_dirs/dotav2_cleanstart/full24e_grouped'
