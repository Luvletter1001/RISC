_base_ = './hrrsd_p15b_oriented_dino_overfit100_4gpu_b8_v5.py'

work_dir = 'work_dirs/p15c_gaussian_quality_hrrsd_20260623/overfit100_4gpu_b8_v5'

model = dict(
    bbox_head=dict(
        type='P15CGaussianQualityDINOSetHead',
        loss_quality_weight=1.0,
        quality_gwd_sigma=4.0))
