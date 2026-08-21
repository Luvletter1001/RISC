default_scope = 'mmrotate'

class_name = ('ship', )
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

data_root = '/data1/zcy/datasets/HRSC2016_DOTA_ship1/'
val_ann_file = 'test/labelTxt/'
val_img_path = 'test/images/'

load_from = '/data1/zcy/OpenRSD/weights/rotated_rtmdet_tiny-9x-hrsc-9f2e3ca6.pth'

test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img_path=val_img_path),
        metainfo=metainfo,
        test_mode=True))
val_dataloader = test_dataloader

val_evaluator = [
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
test_evaluator = val_evaluator
