_base_ = './hrrsd_rtmdet_l_dota_init_internal_epoch3_gpu0189.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.p13e_set_decoder_head'],
    allow_failed_imports=False)

find_unused_parameters = True

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_p14n_rank_quality_e2e_4gpu')

train_cfg = dict(max_epochs=3)

train_dataloader = dict(batch_size=2)

# HRRSD train median rotated width/height in normalized 800x800 coordinates.
# These are query reference boxes, not dense anchors or post-processing rules.
hrrsd_class_wh_priors = [
    [0.40250, 0.38375],  # T
    [0.13625, 0.12125],  # airplane
    [0.46750, 0.40250],  # baseball
    [0.15000, 0.09125],  # basketball
    [0.53250, 0.25625],  # bridge
    [0.43875, 0.41625],  # crossroad
    [0.46375, 0.32125],  # ground
    [0.22125, 0.12250],  # harbor
    [0.25500, 0.11625],  # parking
    [0.25500, 0.10500],  # ship
    [0.18250, 0.16750],  # storage
    [0.14375, 0.10500],  # tennis
    [0.06625, 0.04875],  # vehicle
]

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(
        _delete_=True,
        type='P14NRankQualityQueryTransportHead',
        num_classes=13,
        in_channels=256,
        feat_channels=128,
        num_queries=260,
        num_decoder_layers=2,
        num_heads=8,
        strides=(8, 16, 32),
        image_size=(800, 800),
        support_scale=1.25,
        support_init_std=0.01,
        logit_scale_init=2.0,
        cls_loss_weight=0.75,
        bbox_loss_weight=14.0,
        angle_loss_weight=1.0,
        quality_loss_weight=1.0,
        seed_loss_weight=0.25,
        match_cls_cost=1.0,
        match_bbox_cost=12.0,
        match_angle_cost=1.0,
        query_class_prior_weight=0.50,
        own_class_logit_bias=0.25,
        dn_loss_weight=1.0,
        dn_noise_scale=0.015,
        dn_max_gt=80,
        box_delta_scale=0.65,
        class_wh_priors=hrrsd_class_wh_priors,
        learn_reference_wh=True,
        reference_wh_scale=1.0,
        center_delta_scale=0.45,
        query_transport_loss_weight=1.5,
        query_transport_topk=1,
        teacher_predictions_path=(
            'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
            'teacher_p13c_dense_fdd_obj_train_full/predictions.pkl'),
        teacher_score_thr=0.20,
        teacher_topk_per_img=120,
        teacher_score_power=1.0,
        transport_rank_loss_weight=2.0,
        ranking_margin=0.06,
        rank_quality_floor=0.30,
        rank_quality_center_sigma=0.45,
        rank_quality_size_sigma=0.65,
        rank_quality_angle_sigma=0.60,
        rank_quality_margin_scale=0.12,
        rank_quality_cls_target_blend=1.0,
        score_thr=0.0,
        max_per_img=260))
