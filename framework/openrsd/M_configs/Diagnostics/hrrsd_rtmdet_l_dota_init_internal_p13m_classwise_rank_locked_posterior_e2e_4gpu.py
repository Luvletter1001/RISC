_base_ = './hrrsd_rtmdet_l_dota_init_internal_p13l_class_locked_balanced_posterior_e2e_4gpu.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_p13m_classwise_rank_locked_posterior_e2e_4gpu')

model = dict(
    bbox_head=dict(
        type='P13MClasswiseRankLockedPosteriorHead',
        assigned_rank_loss_weight=5.0,
        ranking_margin=0.10,
        score_thr=0.0,
        per_class_max_per_img=12,
        max_per_img=156))
