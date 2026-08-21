_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_rankdelta_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus2_bass_gsf_p2_posterior_rank')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            consistency_loss=dict(
                enable=False),
            density_head=dict(
                enable=True,
                loss_weight=0.06,
                hidden_channels=96,
                mean_residual_scale=0.50,
                log_std_delta_limit=0.50,
                max_delta_abs=0.25,
                nonpositive_delta=True,
                delta_loss_enable=True,
                delta_loss_weight=0.01,
                delta_loss_min_logprob_gap=1.25,
                delta_loss_min_hardneg_score=0.40,
                delta_loss_min_hardneg_logit=None,
                delta_loss_target_negative_delta=0.15,
                delta_loss_gt_keep_weight=0.25,
                delta_loss_max_hardneg=1,
                log_stats=True))))
