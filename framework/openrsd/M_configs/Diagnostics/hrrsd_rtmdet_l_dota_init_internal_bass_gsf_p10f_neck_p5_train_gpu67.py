_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p10a_fusion_perlevel_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus2_bass_gsf_p10f_neck_p5')

model = dict(
    neck=dict(
        gaussian_support_fusion=dict(
            enabled_levels=[2],
            temperature=0.75,
            gamma_init=0.010,
            residual_scale=0.35,
            zero_init_adapter=False)))
