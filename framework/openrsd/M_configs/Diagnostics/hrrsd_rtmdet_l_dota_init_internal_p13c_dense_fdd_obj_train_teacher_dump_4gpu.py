_base_ = './hrrsd_rtmdet_l_dota_init_internal_p13c_dense_fdd_obj_train_gpu67.py'

data_root = '/data1/zcy/datasets/HRRSD_800_0/internal_split_20260619/'

hrrsd_class_names = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]

metainfo = dict(classes=hrrsd_class_names, palette=[(220, 20, 60)])

teacher_test_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args=dict(backend='disk')),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad',
        size=(800, 800),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'teacher_p13c_dense_fdd_obj_train_full')

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/'),
        img_shape=(800, 800),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=teacher_test_pipeline))

test_evaluator = dict(type='DOTAMetric', metric='mAP')
