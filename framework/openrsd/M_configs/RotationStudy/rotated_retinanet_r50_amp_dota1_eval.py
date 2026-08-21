_base_ = [
    '/data1/zcy/OpenRSD/mmrotate_configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_amp-1x_dota.py',
]

model = dict(backbone=dict(init_cfg=None))

data_root = '/data1/zcy/OpenRSD/data/DOTA1_1024_500/'

train_dataloader = None
train_cfg = None
optim_wrapper = None
param_scheduler = None

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='angle_sweep_val/realistic/angle_000/annfiles/',
        data_prefix=dict(img_path='angle_sweep_val/realistic/angle_000/images/'),
        img_shape=(1024, 1024),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True)))

val_dataloader = test_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

load_from = '/data1/zcy/OpenRSD/weights/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90-01de71b5.pth'
