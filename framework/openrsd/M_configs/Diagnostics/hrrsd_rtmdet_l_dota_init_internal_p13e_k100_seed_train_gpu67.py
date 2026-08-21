_base_ = './hrrsd_rtmdet_l_dota_init_internal_epoch3_gpu0189.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.p13e_set_decoder_head'],
    allow_failed_imports=False)

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus3_p13e_k100_seed')

hrrsd_class_names = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]

train_cfg = dict(max_epochs=3)

model = dict(
    bbox_head=dict(
        _delete_=True,
        type='P13EDenseSeedGaussianSetHead',
        num_classes=len(hrrsd_class_names),
        in_channels=256,
        feat_channels=128,
        num_queries=100,
        num_decoder_layers=2,
        num_heads=8,
        strides=(8, 16, 32),
        image_size=(800, 800),
        support_scale=1.25,
        support_init_std=0.01,
        logit_scale_init=2.0,
        cls_loss_weight=2.0,
        bbox_loss_weight=5.0,
        angle_loss_weight=1.0,
        quality_loss_weight=1.0,
        seed_loss_weight=0.25,
        match_cls_cost=2.0,
        match_bbox_cost=5.0,
        match_angle_cost=1.0,
        score_thr=0.03,
        max_per_img=100))
