_base_ = './ov_capflow_swin-t_hrsc_c2_balanced_5e.py'

model = dict(
    decoder=dict(
        enable_null_reservoir=True,
        null_reservoir_cfg=dict()),
    null_loss_cfg=dict(
        matched_weight=1.0,
        unmatched_weight=1.0,
        mass_weight=0.1,
        gate_order_weight=0.1,
        gate_margin=0.0))
custom_hooks = [
    dict(
        type='FreezeExceptHook',
        trainable_patterns=[
            r'^decoder\.layers\.\d+\.semantic_fusion\.',
            r'^decoder\.null_reservoir\.',
        ])
]
work_dir = 'work_dirs/ov_capflow_hrsc/hrsc_c3_null_5e'
