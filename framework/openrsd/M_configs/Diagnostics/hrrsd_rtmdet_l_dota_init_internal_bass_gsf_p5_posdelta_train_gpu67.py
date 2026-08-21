_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_delta_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus2_bass_gsf_p5_posdelta')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            adapter_nonpositive_delta=False,
            density_head=dict(
                enable=True,
                loss_weight=0.05,
                hidden_channels=64,
                mean_residual_scale=0.75,
                log_std_delta_limit=0.75,
                max_delta_abs=0.30,
                nonpositive_delta=False,
                apply_logit_delta=False,
                logit_delta_source='head',
                delta_loss_enable=True,
                delta_loss_weight=0.01,
                delta_loss_min_logprob_gap=1.0,
                delta_loss_min_hardneg_score=0.30,
                delta_loss_min_hardneg_logit=None,
                delta_loss_target_negative_delta=0.10,
                delta_loss_gt_keep_weight=0.0,
                delta_loss_max_hardneg=1,
                positive_delta_loss_enable=True,
                positive_delta_loss_weight=0.01,
                positive_delta_loss_target=0.10,
                positive_delta_loss_min_gt_logprob=None,
                positive_delta_loss_min_gt_score=0.10,
                log_stats=True))))
