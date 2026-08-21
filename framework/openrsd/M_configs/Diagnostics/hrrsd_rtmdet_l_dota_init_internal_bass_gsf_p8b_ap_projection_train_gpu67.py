_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p7_scalecons_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus2_bass_gsf_p8b_ap_projection')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            level_routing=dict(
                enable=True,
                level_ref_sizes=[48.0, 128.0, 320.0],
                weight=0.20,
                max_bias_abs=0.50,
                target_assignment_source='pre_routing',
                learnable_gate_enable=True,
                learnable_gate_init=0.50,
                projection_loss_enable=True,
                projection_loss_weight=0.02,
                projection_loss_max_drop=0.05,
                projection_loss_min_assign_metric=0.0,
                log_stats=True))))
