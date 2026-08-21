_base_ = '../OfficialMMRotateWeightRepro/kfiou/r3det-oc_r50_fpn_kfiou-ln_1x_dota.py'

load_from = '/data1/zcy/OpenRSD/weights/r3det_kfiou_ln_r50_fpn_1x_dota_oc-8e7f049d_mmrotate1xkeys_no_cls.pth'

num_classes = 18
data_root = '/data1/zcy/OpenRSD/data/DOTA2_1024_500/'

model = dict(
    bbox_head_init=dict(num_classes=num_classes),
    bbox_head_refine=[
        dict(
            type='R3RefineHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            frm_cfg=dict(
                type='FRM', feat_channels=256, strides=[8, 16, 32, 64, 128]),
            anchor_generator=dict(
                type='PseudoRotatedAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version='oc',
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox_type='kfiou',
            loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=5.0))
    ])

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

work_dir = 'work_dirs/dotav2_official_adapters/r3det_kfiou_r50_fpn_fullinit'
