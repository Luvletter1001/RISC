_base_ = './hrrsd_rtmdet_l_dota_init_internal_epoch3_gpu0189.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.sgp_rtmdet_head'],
    allow_failed_imports=False)

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus3_sgp_gaussian_product_posterior_head')

hrrsd_class_names = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]

geometry_priors_csv = (
    'work_dirs/gs3c_dataset_inventory_20260619/'
    'hrrsd_internal_train_geometry_priors.csv')

train_cfg = dict(max_epochs=3)

model = dict(
    bbox_head=dict(
        type='SGPRotatedRTMDetSepBNHead',
        num_classes=len(hrrsd_class_names),
        sgp_head=dict(
            embed_channels=64,
            semantic_weight=1.0,
            geometry_weight=0.025,
            objectness_weight=0.10,
            support_mu_init_std=0.25,
            support_logvar_init=0.0,
            logit_scale_init=6.0,
            query_logvar_min=-6.0,
            query_logvar_max=4.0,
            support_logvar_min=-6.0,
            support_logvar_max=4.0,
            class_names=hrrsd_class_names,
            class_geometry_priors_csv=geometry_priors_csv)))
