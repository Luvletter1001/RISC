_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p8b_ap_projection_eval_gpu67.py'

custom_imports = dict(
    imports=[
        'M_AD.models.dense_heads.gs3c_rtmdet_head',
        'M_AD.models.backbones.gaussian_support_cspnext',
    ],
    allow_failed_imports=False)

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'eval_epoch3_plus2_bass_gsf_p10g_backbone_c5_safe_head_full')

hrrsd_class_names = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]
geometry_priors_csv = (
    'work_dirs/gs3c_dataset_inventory_20260619/'
    'hrrsd_internal_train_geometry_priors.csv')

model = dict(
    backbone=dict(
        type='GaussianSupportCSPNeXt',
        gaussian_support_adapter=dict(
            enable=True,
            adapter_type='global_channel',
            class_geometry_priors_csv=geometry_priors_csv,
            class_names=hrrsd_class_names,
            num_classes=len(hrrsd_class_names),
            stages=(4,),
            token_dim=32,
            temperature=1.0,
            gamma_init=0.005,
            residual_scale=0.25,
            zero_init_adapter=True)))
