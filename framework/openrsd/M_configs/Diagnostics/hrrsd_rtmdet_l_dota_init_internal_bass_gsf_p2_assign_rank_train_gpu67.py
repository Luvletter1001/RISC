_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_rankdelta_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus2_bass_gsf_p2_assign_rank')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            consistency_loss=dict(
                enable=True,
                loss_weight=0.02,
                margin=0.35,
                min_logprob_gap=1.0,
                max_hardneg=1,
                min_hardneg_logit=0.0,
                max_gt_abs_z=3.0,
                log_stats=True),
            density_head=dict(
                enable=True,
                loss_weight=0.05,
                hidden_channels=64,
                mean_residual_scale=0.75,
                log_std_delta_limit=0.75,
                max_delta_abs=0.30,
                nonpositive_delta=True,
                delta_loss_enable=True,
                delta_loss_weight=0.01,
                delta_loss_min_logprob_gap=1.0,
                delta_loss_min_hardneg_score=0.30,
                delta_loss_min_hardneg_logit=None,
                delta_loss_target_negative_delta=0.15,
                delta_loss_gt_keep_weight=0.20,
                delta_loss_max_hardneg=1,
                log_stats=True))))
