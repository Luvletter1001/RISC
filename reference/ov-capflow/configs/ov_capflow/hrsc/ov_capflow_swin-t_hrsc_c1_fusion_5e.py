_base_ = './ov_capflow_swin-t_hrsc_c0_native_10e.py'

model = dict(
    decoder=dict(
        enable_null_reservoir=False,
        layer_cfg=dict(
            enable_semantic_fusion=True,
            enable_density_capacity=False,
            semantic_fusion_cfg=dict(adapter_init='identity'))),
    bbox_head=dict(
        balanced_cfg=dict(
            enabled=False, matched_weight=1.0, unmatched_weight=1.0)))
load_from = 'work_dirs/ov_capflow_hrsc/hrsc_c0_native_10e/epoch_10.pth'
train_cfg = dict(max_epochs=5, val_interval=5)
param_scheduler = [
    dict(
        type='MultiStepLR', begin=0, end=5, by_epoch=True,
        milestones=[4], gamma=0.1),
]
custom_hooks = [
    dict(
        type='FreezeExceptHook',
        trainable_patterns=[
            r'^decoder\.layers\.\d+\.semantic_fusion\.',
        ])
]
work_dir = 'work_dirs/ov_capflow_hrsc/hrsc_c1_fusion_5e'
