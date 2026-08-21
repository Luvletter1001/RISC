_base_ = './hrrsd_rtmdet_l_dota_init_internal_p14c_class_transport_mini.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'mini_p14n_rank_quality')

model = dict(
    bbox_head=dict(
        type='P14NRankQualityQueryTransportHead',
        transport_rank_loss_weight=2.0,
        ranking_margin=0.06,
        rank_quality_floor=0.30,
        rank_quality_center_sigma=0.45,
        rank_quality_size_sigma=0.65,
        rank_quality_angle_sigma=0.60,
        rank_quality_margin_scale=0.12))
