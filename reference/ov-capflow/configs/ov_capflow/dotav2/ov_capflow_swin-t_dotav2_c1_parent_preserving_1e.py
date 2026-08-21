_base_ = './ov_capflow_swin-t_dotav2_c0_native_1e.py'

model = dict(
    decoder=dict(
        layer_cfg=dict(
            enable_semantic_fusion=True,
            semantic_fusion_cfg=dict(adapter_init='identity'))))
load_from = (
    'work_dirs/ov_capflow_dotav2/'
    'dotav2_c0_native_1e_recovery_static/epoch_1.pth')
custom_hooks = [
    dict(
        type='FreezeExceptHook',
        trainable_patterns=[r'^decoder\.layers\.\d+\.semantic_fusion\.'])
]
work_dir = (
    'work_dirs/ov_capflow_dotav2/'
    'dotav2_c1_parent_preserving_1e')
