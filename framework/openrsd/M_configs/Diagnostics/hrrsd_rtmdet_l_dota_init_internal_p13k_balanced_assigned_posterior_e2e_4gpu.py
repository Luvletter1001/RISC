_base_ = './hrrsd_rtmdet_l_dota_init_internal_epoch3_gpu0189.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.p13e_set_decoder_head'],
    allow_failed_imports=False)

find_unused_parameters = True

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_p13k_balanced_assigned_posterior_e2e_4gpu')

hrrsd_class_names = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]

train_cfg = dict(max_epochs=3)

# Base HRRSD config uses DefaultSampler, so only override batch size.
train_dataloader = dict(batch_size=2)

model = dict(
    bbox_head=dict(
        _delete_=True,
        type='P13KBalancedAssignedPosteriorHead',
        num_classes=len(hrrsd_class_names),
        in_channels=256,
        feat_channels=128,
        num_queries=520,
        num_decoder_layers=2,
        num_heads=8,
        strides=(8, 16, 32),
        image_size=(800, 800),
        support_scale=1.25,
        support_init_std=0.01,
        logit_scale_init=2.0,
        cls_loss_weight=2.0,
        bbox_loss_weight=8.0,
        angle_loss_weight=1.0,
        quality_loss_weight=1.0,
        seed_loss_weight=0.5,
        match_cls_cost=2.0,
        match_bbox_cost=8.0,
        match_angle_cost=1.0,
        query_class_prior_weight=0.75,
        own_class_logit_bias=0.40,
        dn_loss_weight=1.0,
        dn_noise_scale=0.015,
        dn_max_gt=80,
        box_delta_scale=0.75,
        center_delta_scale=0.35,
        seed_gaussian_sigma=0.045,
        quality_center_sigma=0.16,
        per_class_query_layout=True,
        ranking_loss_weight=1.5,
        ranking_margin=0.10,
        aux_recall_loss_weight=0.0,
        aux_box_loss_weight=5.0,
        aux_box_angle_loss_weight=1.0,
        aux_box_min_weight=0.0,
        assigned_lattice_loss_weight=1.0,
        assigned_lattice_sigma=0.15,
        assigned_lattice_min_weight=0.03,
        assigned_lattice_topk_floor=0.50,
        assigned_topk_per_gt=4,
        assigned_rank_loss_weight=5.0,
        assigned_positive_quality_gain=5.0,
        size_prior_init=(0.055, 0.055),
        size_prior_mix=0.55,
        size_prior_sigma=0.80,
        size_likelihood_weight=0.0,
        positive_quality_floor=0.75,
        quality_size_sigma=0.65,
        quality_angle_sigma=0.75,
        score_thr=0.002,
        max_per_img=100))
