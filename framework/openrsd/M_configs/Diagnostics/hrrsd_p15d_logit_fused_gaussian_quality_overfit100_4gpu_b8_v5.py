_base_ = './hrrsd_p15c_gaussian_quality_overfit100_4gpu_b8_v5.py'

work_dir = 'work_dirs/p15d_logit_fused_gaussian_quality_hrrsd_20260623/overfit100_4gpu_b8_v5'

model = dict(
    bbox_head=dict(
        type='P15DLogitFusedGaussianQualityDINOSetHead',
        quality_logit_alpha=0.5,
        learnable_quality_logit_alpha=False))
