_base_ = './hrrsd_rtmdet_l_dota_init_internal_epoch3_gpu0189.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.p13e_set_decoder_head'],
    allow_failed_imports=False)

find_unused_parameters = True

max_iters = 500
train_subset = 20
eval_subset = 128

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'mini_p14a_query_transport')

hrrsd_class_names = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=max_iters + 1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0, by_epoch=False, begin=0, end=max_iters)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=max_iters,
        save_last=True,
        max_keep_ckpts=2))

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(indices=train_subset))

val_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(indices=eval_subset))

test_dataloader = val_dataloader

model = dict(
    bbox_head=dict(
        _delete_=True,
        type='P14RotatedDINOQueryTransportHead',
        num_classes=len(hrrsd_class_names),
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
        cls_loss_weight=2.0,
        bbox_loss_weight=8.0,
        angle_loss_weight=1.0,
        quality_loss_weight=1.0,
        seed_loss_weight=0.25,
        match_cls_cost=2.0,
        match_bbox_cost=8.0,
        match_angle_cost=1.0,
        query_class_prior_weight=0.50,
        own_class_logit_bias=0.25,
        dn_loss_weight=1.0,
        dn_noise_scale=0.015,
        dn_max_gt=80,
        box_delta_scale=0.75,
        query_transport_loss_weight=1.0,
        query_transport_topk=1,
        teacher_predictions_path=(
            'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
            'teacher_p13c_dense_fdd_obj_train_full/predictions.pkl'),
        teacher_score_thr=0.20,
        teacher_topk_per_img=120,
        teacher_score_power=1.0,
        score_thr=0.0,
        max_per_img=260))
