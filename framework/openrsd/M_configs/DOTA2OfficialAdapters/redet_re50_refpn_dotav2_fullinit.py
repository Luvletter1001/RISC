_base_ = '../OfficialMMRotateWeightRepro/redet/redet-le90_re50_refpn_1x_dota.py'

load_from = '/data1/zcy/OpenRSD/weights/redet_re50_fpn_1x_dota_le90-724ab2da_no_cls.pth'

num_classes = 18
data_root = '/data1/zcy/OpenRSD/data/DOTA2_1024_500/'

model = dict(
    # The official ReDet config points to a missing external ReResNet
    # pretrain file. The detector checkpoint below already contains backbone
    # weights, so disable that separate init path.
    backbone=dict(init_cfg=None),
    roi_head=dict(
        bbox_head=[
            dict(
                type='mmdet.Shared2FCBBoxHead',
                predict_box_type='rbox',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                bbox_coder=dict(
                    type='DeltaXYWHTHBBoxCoder',
                    angle_version='le90',
                    norm_factor=2,
                    edge_swap=True,
                    target_means=(.0, .0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.2, 0.2, 0.1),
                    use_box_type=True),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='mmdet.SmoothL1Loss',
                    beta=1.0,
                    loss_weight=1.0)),
            dict(
                type='mmdet.Shared2FCBBoxHead',
                predict_box_type='rbox',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                bbox_coder=dict(
                    type='DeltaXYWHTRBBoxCoder',
                    angle_version='le90',
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1, 0.05]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='mmdet.SmoothL1Loss',
                    beta=1.0,
                    loss_weight=1.0))
        ]))

train_dataloader = dict(
    dataset=dict(
        type='DOTAv2Dataset',
        data_root=data_root,
        ann_file='ss_train/annfiles/',
        data_prefix=dict(img_path='ss_train/images/')))

val_dataloader = dict(
    dataset=dict(
        type='DOTAv2Dataset',
        data_root=data_root,
        ann_file='ss_val/annfiles/',
        data_prefix=dict(img_path='ss_val/images/')))

test_dataloader = val_dataloader

work_dir = 'work_dirs/dotav2_official_adapters/redet_re50_refpn_fullinit'
