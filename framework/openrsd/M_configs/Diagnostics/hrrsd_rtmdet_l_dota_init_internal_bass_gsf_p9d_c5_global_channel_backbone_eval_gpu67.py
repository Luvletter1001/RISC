_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p9c_global_channel_backbone_eval_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'eval_epoch3_plus2_bass_gsf_p9d_c5_global_channel_backbone_head_full')

model = dict(
    backbone=dict(
        gaussian_support_adapter=dict(
            adapter_type='global_channel',
            stages=(4,),
            token_dim=32,
            temperature=0.75,
            gamma_init=0.03,
            residual_scale=0.5,
            zero_init_adapter=True)))
