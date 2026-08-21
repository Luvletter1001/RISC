_base_ = './hrrsd_rtmdet_l_dota_init_internal_p14c_class_transport_mini.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'mini_p14e_quality_transport')

model = dict(
    bbox_head=dict(
        type='P14EQualityCalibratedQueryTransportHead',
        geometry_quality_loss_weight=1.5,
        geometry_quality_cost_scale=0.75))
