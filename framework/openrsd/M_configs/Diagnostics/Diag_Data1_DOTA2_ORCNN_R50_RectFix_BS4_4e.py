_base_ = '../G02_Baselines/Data1_DOTA2/G02_Baselines_Data1_DOTA2_M5_ORCNN_R50.py'

# Short diagnostic: isolate the rect_obj_labels bug while keeping the
# original R50/800px/batch-4 setup comparable to the bad baseline.
max_epochs = 4
val_interval = 4
ckpt_interval = 4

train_cfg = dict(max_epochs=max_epochs, val_interval=val_interval)
default_hooks = dict(checkpoint=dict(interval=ckpt_interval))

file_client_args = dict(backend='disk')
img_scale = (800, 800)

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
    batch_size=4,
    dataset=dict(
        ann_file='ss_train/annfiles/',
        data_prefix=dict(img_path='ss_train/images/'),
        img_shape=img_scale,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='ss_val/annfiles/',
        data_prefix=dict(img_path='ss_val/images/'),
        img_shape=img_scale,
        pipeline=val_pipeline))

test_dataloader = val_dataloader

work_dir = 'work_dirs/diagnostics/orcnn_r50_rectfix_bs4_4e'
