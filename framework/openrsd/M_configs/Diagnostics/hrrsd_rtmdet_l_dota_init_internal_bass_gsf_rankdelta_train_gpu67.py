_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_delta_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus2_bass_gsf_rankdelta_dw005_ms050')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            density_head=dict(
                delta_loss_weight=0.05,
                delta_loss_min_hardneg_score=0.50,
                delta_loss_min_hardneg_logit=None,
                delta_loss_target_negative_delta=0.25,
                delta_loss_gt_keep_weight=0.05,
                delta_loss_max_hardneg=1,
                log_stats=True))))
