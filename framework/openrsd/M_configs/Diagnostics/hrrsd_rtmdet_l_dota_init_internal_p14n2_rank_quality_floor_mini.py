_base_ = './hrrsd_rtmdet_l_dota_init_internal_p14n_rank_quality_mini.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'mini_p14n2_rank_quality_floor')

model = dict(
    bbox_head=dict(
        rank_quality_floor=0.55,
        rank_quality_margin_scale=0.08))
