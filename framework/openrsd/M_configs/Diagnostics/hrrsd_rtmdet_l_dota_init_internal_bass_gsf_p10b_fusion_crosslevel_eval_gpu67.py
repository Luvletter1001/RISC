_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p8b_ap_projection_eval_gpu67.py'

custom_imports = dict(
    imports=[
        'M_AD.models.dense_heads.gs3c_rtmdet_head',
        'M_AD.models.necks.gaussian_support_pafpn',
    ],
    allow_failed_imports=False)

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'eval_epoch3_plus2_bass_gsf_p10b_fusion_crosslevel_head_full')

hrrsd_class_names = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]
geometry_priors_csv = (
    'work_dirs/gs3c_dataset_inventory_20260619/'
    'hrrsd_internal_train_geometry_priors.csv')

model = dict(
    neck=dict(
        type='GaussianSupportCSPNeXtPAFPN',
        gaussian_support_fusion=dict(
            enable=True,
            mode='cross_level_channel',
            class_geometry_priors_csv=geometry_priors_csv,
            class_names=hrrsd_class_names,
            num_classes=len(hrrsd_class_names),
            token_dim=32,
            temperature=0.75,
            gamma_init=0.005,
            residual_scale=0.25,
            level_context_scale=1.0,
            zero_init_adapter=False)))
