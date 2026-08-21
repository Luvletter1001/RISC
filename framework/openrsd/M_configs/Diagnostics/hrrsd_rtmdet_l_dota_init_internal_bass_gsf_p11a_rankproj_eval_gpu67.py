_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p8b_ap_projection_eval_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'eval_epoch3_plus3_bass_gsf_p11a_ap_safe_ranking_projection_head_full')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            adapter_nonpositive_delta=False,
            geometry_support_enable=True,
            class_geometry_priors_csv=(
                'work_dirs/gs3c_dataset_inventory_20260619/'
                'hrrsd_internal_train_geometry_priors.csv'),
            density_head=dict(
                enable=True,
                use_geometry_logprob=True,
                nonpositive_delta=False,
                apply_logit_delta=False,
                delta_loss_enable=False,
                positive_delta_loss_enable=False,
                pair_margin_loss_enable=False,
                support_negative_loss_enable=False,
                scale_consistency_loss_enable=False,
                log_stats=True),
            level_routing=dict(
                enable=True,
                level_ref_sizes=[48.0, 128.0, 320.0],
                weight=0.10,
                max_bias_abs=0.25,
                target_assignment_source='pre_routing',
                learnable_gate_enable=True,
                learnable_gate_init=0.50,
                projection_loss_enable=False,
                projection_loss_weight=0.0,
                projection_loss_max_drop=0.03,
                projection_loss_min_assign_metric=0.0,
                log_stats=True),
            ap_safe_support_projection=dict(
                enable=False,
                loss_weight=0.0,
                use_geometry_logprob=True,
                log_stats=False),
            ap_safe_ranking_projection=dict(
                enable=False,
                loss_weight=0.0,
                use_geometry_logprob=True,
                log_stats=False))))

