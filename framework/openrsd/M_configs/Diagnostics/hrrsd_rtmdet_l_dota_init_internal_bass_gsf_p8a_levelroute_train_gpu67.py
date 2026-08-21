_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p7_scalecons_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus2_bass_gsf_p8a_levelroute')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            level_routing=dict(
                enable=True,
                level_ref_sizes=[48.0, 128.0, 320.0],
                weight=0.10,
                max_bias_abs=0.50))))
