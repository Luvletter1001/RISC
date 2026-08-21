_base_ = './hrrsd_rtmdet_l_dota_init_internal_epoch3_gpu0189.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.p13e_set_decoder_head'],
    allow_failed_imports=False)

find_unused_parameters = True

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_p13i_lattice_posterior_e2e_4gpu')

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
        type='P13IStratifiedLatticePosteriorHead',
        num_classes=len(hrrsd_class_names),
        in_channels=256,
        feat_channels=128,
        num_queries=390,
        num_decoder_layers=2,
        num_heads=8,
        strides=(8, 16, 32),
        image_size=(800, 800),
        support_scale=1.25,
        support_init_std=0.01,
        logit_scale_init=1.5,
        cls_loss_weight=2.0,
        bbox_loss_weight=8.0,
        angle_loss_weight=1.0,
        quality_loss_weight=1.0,
        seed_loss_weight=0.5,
        match_cls_cost=2.0,
        match_bbox_cost=8.0,
        match_angle_cost=1.0,
        query_class_prior_weight=0.75,
        own_class_logit_bias=0.35,
        dn_loss_weight=1.0,
        dn_noise_scale=0.015,
        dn_max_gt=80,
        box_delta_scale=0.65,
        seed_gaussian_sigma=0.045,
        quality_center_sigma=0.16,
        per_class_query_layout=True,
        ranking_loss_weight=4.0,
        ranking_margin=0.05,
        aux_recall_loss_weight=0.35,
        aux_recall_sigma=0.12,
        aux_box_loss_weight=4.0,
        aux_box_angle_loss_weight=1.0,
        aux_box_min_weight=0.03,
        size_prior_init=(0.055, 0.055),
        size_prior_mix=0.55,
        size_prior_sigma=0.80,
        size_likelihood_weight=0.0,
        positive_quality_floor=0.70,
        quality_size_sigma=0.65,
        quality_angle_sigma=0.75,
        score_thr=0.0025,
        max_per_img=80))
