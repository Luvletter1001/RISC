_base_ = './hrrsd_rtmdet_l_dota_init_internal_p13k_balanced_assigned_posterior_e2e_4gpu.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_p13l_class_locked_balanced_posterior_e2e_4gpu')

model = dict(
    bbox_head=dict(
        type='P13LClassLockedBalancedPosteriorHead',
        score_thr=0.0,
        per_class_max_per_img=12,
        max_per_img=156))
