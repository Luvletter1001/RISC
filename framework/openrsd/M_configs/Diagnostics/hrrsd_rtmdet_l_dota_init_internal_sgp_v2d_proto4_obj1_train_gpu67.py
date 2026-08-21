_base_ = './hrrsd_rtmdet_l_dota_init_internal_sgp_head_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus3_sgp_v2d_proto025_obj075')

hrrsd_class_names = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]

geometry_priors_csv = (
    'work_dirs/gs3c_dataset_inventory_20260619/'
    'hrrsd_internal_train_geometry_priors.csv')

model = dict(
    bbox_head=dict(
        sgp_head=dict(
            embed_channels=64,
            semantic_weight=1.0,
            geometry_weight=0.005,
            objectness_weight=0.75,
            prototype_cosine_weight=0.25,
            support_mu_init='orthogonal',
            support_mu_scale=1.0,
            support_mu_init_std=0.0,
            support_logvar_init=0.0,
            query_logvar_bias_init=0.0,
            logit_scale_init=6.0,
            class_names=hrrsd_class_names,
            class_geometry_priors_csv=geometry_priors_csv)))
