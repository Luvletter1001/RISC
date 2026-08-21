_base_ = './hrrsd_rtmdet_l_dota_init_internal_p14c_class_transport_mini.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'mini_p14i_gaussian_reference')

model = dict(
    bbox_head=dict(
        type='P14IGaussianReferenceQueryTransportHead',
        gaussian_geometry_weight=0.50,
        gaussian_geometry_init_std=0.01))
