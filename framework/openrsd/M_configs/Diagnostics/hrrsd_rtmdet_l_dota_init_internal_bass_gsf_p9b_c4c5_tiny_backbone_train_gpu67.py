_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p9a_gst_backbone_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus2_bass_gsf_p9b_c4c5_tiny_backbone')

model = dict(
    backbone=dict(
        gaussian_support_adapter=dict(
            adapter_type='spatial_gate',
            stages=(3, 4),
            token_dim=32,
            temperature=1.0,
            gamma_init=0.005,
            residual_scale=0.25)))
