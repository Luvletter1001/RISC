_base_ = './hrrsd_rtmdet_l_dota_init_internal_p14c_class_transport_mini.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'mini_p14l_gaussian_geometry')

model = dict(
    bbox_head=dict(
        type='P14LGaussianGeometryQueryTransportHead',
        gwd_loss_weight=4.0,
        gwd_alpha=1.0,
        gwd_normalize=True))
