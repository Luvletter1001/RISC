_base_ = '../G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M9_RTMDet_M.py'

data_root = '/data1/zcy/datasets/xView_New_800_600/'

class_name = [
    'Aircraft_Hangar', 'Barge', 'Building', 'Bus', 'Cargo_Truck',
    'Cargo_or_Container_Car', 'Cement_Mixer', 'Construction_Site',
    'Container_Crane', 'Container_Ship', 'Crane_Truck', 'Damaged_Building',
    'Dump_Truck', 'Engineering_Vehicle', 'Excavator', 'Facility', 'Ferry',
    'Fishing_Vessel', 'Fixed-wing_Aircraft', 'Flat_Car',
    'Front_loader_or_Bulldozer', 'Ground_Grader', 'Haul_Truck',
    'Helicopter', 'Helipad', 'Hut_or_Tent', 'Locomotive',
    'Maritime_Vessel', 'Mobile_Crane', 'Motorboat', 'Oil_Tanker',
    'Passenger_Car', 'Passenger_Vehicle', 'Passenger_or_Cargo_Plane',
    'Pickup_Truck', 'Pylon', 'Railway_Vehicle', 'Reach_Stacker', 'Sailboat',
    'Scraper_or_Tractor', 'Shed', 'Shipping_Container',
    'Shipping_container_lot', 'Small_Aircraft', 'Small_Car', 'Storage_Tank',
    'Straddle_Carrier', 'Tank_car', 'Tower', 'Tower_crane', 'Trailer',
    'Truck', 'Truck_Tractor', 'Truck_Tractor_with_Box_Trailer',
    'Truck_Tractor_with_Flatbed_Trailer',
    'Truck_Tractor_with_Liquid_Tank', 'Tugboat', 'Utility_Truck',
    'Vehicle_Lot', 'Yacht'
]
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])
num_classes = len(class_name)

max_iters = 20
train_subset = 256
eval_subset = 64

work_dir = 'work_dirs/gs3c_xview_network_smoke_20260619/baseline_smoke20'

load_from = None
resume = False

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=max_iters + 1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0, by_epoch=False, begin=0, end=max_iters)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=max_iters,
        save_last=True,
        max_keep_ckpts=2))

custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_labeled/labelTxt/',
        data_prefix=dict(img_path='train_labeled/images/'),
        img_shape=(800, 800),
        filter_cfg=dict(filter_empty_gt=True),
        indices=train_subset))

val_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/annfiles/',
        data_prefix=dict(img_path='test/images/'),
        img_shape=(800, 800),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True),
        indices=eval_subset))

test_dataloader = val_dataloader

model = dict(
    backbone=dict(norm_cfg=dict(type='BN'), init_cfg=None),
    neck=dict(norm_cfg=dict(type='BN')),
    bbox_head=dict(num_classes=num_classes, norm_cfg=dict(type='BN')))
