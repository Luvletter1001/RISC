_base_ = './grounding_dino_swin-t_hrsc_parent_10e.py'

custom_imports = dict(
    imports=['projects.OVCapFlow.ov_capflow'], allow_failed_imports=False)
model = dict(
    type='OVCapFlow',
    decoder=dict(
        enable_null_reservoir=False,
        layer_cfg=dict(
            enable_semantic_fusion=False,
            enable_density_capacity=False)),
    bbox_head=dict(
        type='OVCapFlowHead',
        balanced_cfg=dict(enabled=False),
        readout_cfg=dict(temperature=1.0, power=1.0, use_capacity=False)),
    density_loss_cfg=dict(weight=0.0),
    null_loss_cfg=dict(),
    test_cfg=dict(_delete_=True))
work_dir = 'work_dirs/ov_capflow_hrsc/hrsc_c0_native_10e'
