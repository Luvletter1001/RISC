_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p6_supportneg_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus2_bass_gsf_p7_scalecons')

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
                loss_weight=0.05,
                hidden_channels=64,
                mean_residual_scale=0.75,
                log_std_delta_limit=0.75,
                max_delta_abs=0.25,
                nonpositive_delta=False,
                apply_logit_delta=False,
                logit_delta_source='head',
                use_geometry_logprob=True,
                delta_loss_enable=False,
                positive_delta_loss_enable=False,
                pair_margin_loss_enable=False,
                support_negative_loss_enable=True,
                support_negative_loss_weight=0.005,
                support_negative_loss_min_score=0.05,
                support_negative_loss_min_logprob_gap=1.5,
                support_negative_loss_gamma=2.0,
                support_negative_loss_gap_scale=8.0,
                support_negative_loss_max_extra_weight=1.0,
                support_negative_loss_max_hardneg=1,
                scale_consistency_loss_enable=True,
                scale_consistency_loss_weight=0.01,
                scale_consistency_loss_max_logprob_drop=0.5,
                scale_consistency_loss_min_target_logprob=None,
                log_stats=True))))
