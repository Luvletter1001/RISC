_base_ = './hrrsd_rtmdet_l_dota_init_internal_p14c_class_transport_mini.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'mini_p14m_iterative_refine')

model = dict(
    bbox_head=dict(
        type='P14MIterativeRefineQueryTransportHead',
        intermediate_refine_loss_weight=0.5,
        iterative_center_delta_scale=0.20,
        iterative_angle_delta_scale=0.7853981633974483,
        detach_refine_between_layers=True))
