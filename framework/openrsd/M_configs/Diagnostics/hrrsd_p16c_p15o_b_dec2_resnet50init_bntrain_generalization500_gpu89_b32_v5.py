_base_ = './hrrsd_p16b_p15o_b_dec2_resnet50init_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16c_p15o_b_dec2_resnet50init_bntrain_generalization500_gpu89_b32_v5')

# Prepared fallback if P16B shows that ImageNet init alone is insufficient.
# This tests the BN-specific hypothesis: HRRSD statistics may be too far from
# ImageNet/frozen BN statistics. Batch is 32/GPU, so per-GPU BN estimates are
# usable without switching the whole ResNet to SyncBN.
model = dict(
    backbone=dict(
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False))
