_base_ = './hrrsd_rtmdet_l_dota_init_internal_p13k_balanced_assigned_posterior_e2e_4gpu.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.p13e_set_decoder_head'],
    allow_failed_imports=False)

find_unused_parameters = True

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_p13n_teacher_anchored_posterior_e2e_4gpu')

train_cfg = dict(max_epochs=3)

train_dataloader = dict(batch_size=2)

model = dict(
    bbox_head=dict(
        type='P13NTeacherAnchoredPosteriorHead',
        teacher_predictions_path=(
            'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
            'teacher_p13c_dense_fdd_obj_train_full/predictions.pkl'),
        teacher_loss_weight=0.75,
        teacher_rank_loss_weight=3.0,
        teacher_gaussian_sigma=0.12,
        teacher_score_thr=0.20,
        teacher_topk_per_img=120,
        teacher_score_power=1.0,
        teacher_min_weight=0.03,
        assigned_rank_loss_weight=4.0,
        ranking_margin=0.10,
        assigned_positive_quality_gain=5.0,
        center_delta_scale=0.35,
        score_thr=0.0,
        max_per_img=520))
