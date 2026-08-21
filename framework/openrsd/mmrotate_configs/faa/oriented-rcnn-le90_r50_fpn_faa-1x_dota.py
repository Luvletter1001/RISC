_base_ = ['../oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py']

custom_imports = dict(
    imports=[
        'mmrotate.models.necks.faafusion',
        'mmrotate.models.roi_heads.bbox_heads.faa_head',
    ],
    allow_failed_imports=False)

find_unused_parameters = True

model = dict(
    neck=dict(
        type='FAAFusionFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        fusion_modes=['add', 'add', 'faa'],
        fam_cfg=dict(m=7, c_mid=64)),
    roi_head=dict(
        bbox_head=dict(
            type='FAAHead',
            predict_box_type='rbox',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            cls_predictor_cfg=dict(type='mmdet.Linear'))))

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))
