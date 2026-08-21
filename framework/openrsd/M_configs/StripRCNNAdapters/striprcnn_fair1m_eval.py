_base_ = [
    '/data1/zcy/OpenRSD/mmrotate_configs/_base_/datasets/dota.py',
    '/data1/zcy/OpenRSD/mmrotate_configs/_base_/schedules/schedule_3x.py',
    '/data1/zcy/OpenRSD/mmrotate_configs/_base_/default_runtime.py'
]

angle_version = 'le90'
gpu_number = 8

load_from = '/data1/zcy/OpenRSD/weights/stripnet_s_fair1m.pth'

class_name = [
    'a220', 'a321', 'a330', 'a350', 'arj21',
    'baseball_field', 'basketball_court', 'boeing737', 'boeing747', 'boeing777', 'boeing787',
    'bridge', 'bus', 'c919', 'cargo_truck', 'dry_cargo_ship', 'dump_truck',
    'engineering_ship', 'excavator', 'fishing_boat', 'football_field', 'intersection',
    'liquid_cargo_ship', 'motorboat', 'other-airplane', 'other-ship', 'other-vehicle',
    'passenger_ship', 'roundabout',
    'small_car', 'tennis_court', 'tractor', 'trailer', 'truck_tractor', 'tugboat', 'van', 'warship'
]
metainfo = dict(classes=class_name)
num_classes = len(class_name)

model = dict(
    type='StripRCNN',
    backbone=dict(
        type='StripNet',
        embed_dims=[64, 128, 320, 512],
        k1s=[1, 1, 1, 1],
        k2s=[19, 19, 19, 19],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[2, 2, 4, 2],
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2, clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='StripHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=0, pos_weight=-1, debug=False),
        rpn_proposal=dict(nms_pre=2000, max_per_img=2000, nms=dict(type='nms', iou_threshold=0.8), min_bbox_size=0),
        rcnn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5, match_low_quality=False, iou_calculator=dict(type='RBboxOverlaps2D'), ignore_iof_thr=-1),
            sampler=dict(type='RRandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1, debug=False)),
    test_cfg=dict(
        rpn=dict(nms_pre=2000, max_per_img=2000, nms=dict(type='nms', iou_threshold=0.8), min_bbox_size=0),
        rcnn=dict(nms_pre=2000, min_bbox_size=0, score_thr=0.05, nms=dict(iou_thr=0.1), max_per_img=2000)))

data_root = '/data1/zcy/OpenRSD/data/fair1m/dair1m_1024/'

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        img_shape=(1024, 1024),
        filter_cfg=dict(filter_empty_gt=True)))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        img_shape=(1024, 1024),
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='mmdet.PackDetInputs',
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ]))

test_dataloader = val_dataloader
val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

train_dataloader = None
train_cfg = None
optim_wrapper = None
param_scheduler = None
