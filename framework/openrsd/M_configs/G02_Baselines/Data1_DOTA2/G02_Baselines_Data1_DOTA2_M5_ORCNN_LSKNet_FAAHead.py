_base_ = ['./G02_Baselines_Data1_DOTA2_M5_ORCNN_R50.py']

custom_imports = dict(
    imports=[
        'M_AD.datasets.dota_clamp',
        'mmrotate.models.backbones.lsknet',
        'mmrotate.models.roi_heads.bbox_heads.faa_head',
    ],
    allow_failed_imports=False)

find_unused_parameters = True

img_scale = (1024, 1024)
train_batch_size = 2
num_classes = 18
file_client_args = dict(backend='disk')

train_ann_file = 'ss_train/annfiles/'
train_img_path = 'ss_train/images/'
val_ann_file = 'ss_val/annfiles/'
val_img_path = 'ss_val/images/'

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=[11, 15]),
    dict(
        type='mmdet.Pad',
        size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad',
        size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size=img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(
        ann_file=train_ann_file,
        data_prefix=dict(img_path=train_img_path),
        img_shape=img_scale,
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        ann_file=val_ann_file,
        data_prefix=dict(img_path=val_img_path),
        img_shape=img_scale,
        pipeline=val_pipeline))

test_dataloader = val_dataloader

model = dict(
    backbone=dict(
        _delete_=True,
        type='LSKNet',
        embed_dims=[64, 128, 320, 512],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[2, 2, 4, 2],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/data1/zcy/OpenRSD/weights/lsk_s_fpn_1x_dota_le90_20230116-99749191.pth',
            prefix='backbone.'),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5),
    roi_head=dict(bbox_head=dict(type='FAAHead', num_classes=num_classes)))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0002,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2))

work_dir = 'work_dirs/faahead_dotav2_ss_lsknet_bs2'
