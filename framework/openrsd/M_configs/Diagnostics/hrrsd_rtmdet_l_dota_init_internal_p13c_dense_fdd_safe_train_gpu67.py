_base_ = './hrrsd_rtmdet_l_dota_init_internal_epoch3_gpu0189.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.p13_fusion_decoder_head'],
    allow_failed_imports=False)

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus3_p13c_dense_fdd_safe')

hrrsd_class_names = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]

train_cfg = dict(max_epochs=3)

model = dict(
    bbox_head=dict(
        type='P13FusionDecoderRotatedRTMDetSepBNHead',
        num_classes=len(hrrsd_class_names),
        p13_decoder=dict(
            embed_channels=64,
            support_init='orthogonal',
            support_scale=1.5,
            support_init_std=0.02,
            cls_weight=1.0,
            objectness_weight=0.25,
            bbox_delta_weight=0.02,
            angle_delta_weight=0.02,
            logit_scale_init=6.0)))
