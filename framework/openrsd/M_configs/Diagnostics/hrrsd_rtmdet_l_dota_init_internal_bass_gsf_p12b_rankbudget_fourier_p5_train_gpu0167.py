_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p11a_rankproj_train_gpu67.py'

find_unused_parameters = True

custom_imports = dict(
    imports=[
        'M_AD.models.dense_heads.gs3c_rtmdet_head',
        'M_AD.models.necks.fourier_support_pafpn',
    ],
    allow_failed_imports=False)

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus3_bass_gsf_p12b_rankbudget_fourier_gated_gaussian_p5')

hrrsd_class_names = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]
geometry_priors_csv = (
    'work_dirs/gs3c_dataset_inventory_20260619/'
    'hrrsd_internal_train_geometry_priors.csv')

model = dict(
    neck=dict(
        type='FourierSupportCSPNeXtPAFPN',
        fourier_support_fusion=dict(
            enable=True,
            mode='fourier_gated_gaussian',
            enabled_levels=[2],
            fft_size=16,
            hidden_dim=32,
            confidence_temperature=1.0,
            gaussian_support_fusion=dict(
                class_geometry_priors_csv=geometry_priors_csv,
                class_names=hrrsd_class_names,
                num_classes=len(hrrsd_class_names),
                token_dim=32,
                temperature=0.75,
                gamma_init=0.010,
                residual_scale=0.30,
                zero_init_adapter=False))))
