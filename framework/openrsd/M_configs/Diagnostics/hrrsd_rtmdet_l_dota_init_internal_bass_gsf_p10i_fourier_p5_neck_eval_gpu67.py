_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p8b_ap_projection_eval_gpu67.py'

custom_imports = dict(
    imports=[
        'M_AD.models.dense_heads.gs3c_rtmdet_head',
        'M_AD.models.necks.fourier_support_pafpn',
    ],
    allow_failed_imports=False)

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'eval_epoch3_plus2_bass_gsf_p10i_fourier_p5_neck_head_full')

model = dict(
    neck=dict(
        type='FourierSupportCSPNeXtPAFPN',
        fourier_support_fusion=dict(
            enable=True,
            mode='fourier_channel',
            enabled_levels=[2],
            fft_size=16,
            hidden_dim=32,
            gamma_init=0.010,
            residual_scale=0.25,
            max_delta_norm_ratio=0.020,
            zero_init_adapter=False)))
