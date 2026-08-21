_base_ = './hrrsd_rtmdet_l_dota_init_internal_p14c_class_transport_mini.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'mini_p14g_gaussian_seed_objectness')

model = dict(
    bbox_head=dict(
        type='P14GGaussianSeedObjectnessQueryTransportHead',
        seed_gaussian_sigma=0.06,
        seed_score_power=0.50,
        seed_loss_weight=0.75))
