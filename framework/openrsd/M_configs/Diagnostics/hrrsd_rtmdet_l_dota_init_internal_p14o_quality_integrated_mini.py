_base_ = './hrrsd_rtmdet_l_dota_init_internal_p14n_rank_quality_mini.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'mini_p14o_quality_integrated')

model = dict(
    bbox_head=dict(
        type='P14OQualityIntegratedClassPosteriorHead',
        rank_quality_cls_target_blend=1.0))
