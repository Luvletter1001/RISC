_base_ = '../../../mmrotate_configs/rotated_rtmdet/rotated_rtmdet_tiny-9x-hrsc.py'

hrsc_ship_metainfo = dict(
    classes=('ship', ),
    palette=[(220, 20, 60)],
)

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root='/data1/zcy/OpenRSD/data/HRSC_unzip/dota',
        ann_file='ss_val/annfiles/',
        data_prefix=dict(img_path='ss_val/images/'),
        img_shape=(1024, 1024),
        metainfo=hrsc_ship_metainfo,
        test_mode=True,
        pipeline=_base_.val_pipeline))

val_dataloader = test_dataloader

test_evaluator = [
    dict(
        type='DOTAMetric',
        eval_mode='11points',
        prefix='dota_ap07',
        metric='mAP'),
    dict(
        type='DOTAMetric',
        eval_mode='area',
        prefix='dota_ap12',
        metric='mAP'),
]

val_evaluator = test_evaluator
