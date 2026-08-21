_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p8b_ap_projection_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus2_bass_gsf_p8c_apsafe_support_projection')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            level_routing=dict(
                enable=True,
                level_ref_sizes=[48.0, 128.0, 320.0],
                weight=0.10,
                max_bias_abs=0.25,
                target_assignment_source='pre_routing',
                learnable_gate_enable=True,
                learnable_gate_init=0.50,
                projection_loss_enable=True,
                projection_loss_weight=0.02,
                projection_loss_max_drop=0.05,
                projection_loss_min_assign_metric=0.0,
                log_stats=True),
            ap_safe_support_projection=dict(
                enable=True,
                loss_weight=0.015,
                min_logprob_gap=1.5,
                protect_margin=0.20,
                rank_margin=0.05,
                margin_temperature=0.25,
                support_temperature=1.0,
                min_hardneg_score=0.05,
                min_hardneg_logit=None,
                max_hardneg=1,
                max_budget=1.0,
                assign_metric_power=1.0,
                detach_gt_logit=True,
                use_geometry_logprob=True,
                log_stats=True))))
