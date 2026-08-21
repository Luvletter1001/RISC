_base_ = './hrrsd_rtmdet_l_dota_init_internal_p14e_quality_transport_mini.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'mini_p14f_detached_quality')

model = dict(
    bbox_head=dict(
        type='P14FDetachedQualityQueryTransportHead',
        detach_quality_features=True))
