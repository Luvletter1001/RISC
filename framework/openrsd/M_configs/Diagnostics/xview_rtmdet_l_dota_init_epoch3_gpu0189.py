_base_ = '../G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M10_RTMDet_L.py'

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
    'Pickup_Truck', 'Pylon', 'Railway_Vehicle', 'Reach_Stacker',
    'Sailboat', 'Scraper_or_Tractor', 'Shed', 'Shipping_Container',
    'Shipping_container_lot', 'Small_Aircraft', 'Small_Car',
    'Storage_Tank', 'Straddle_Carrier', 'Tank_car', 'Tower',
    'Tower_crane', 'Trailer', 'Truck', 'Truck_Tractor',
    'Truck_Tractor_with_Box_Trailer',
    'Truck_Tractor_with_Flatbed_Trailer',
    'Truck_Tractor_with_Liquid_Tank', 'Tugboat', 'Utility_Truck',
    'Vehicle_Lot', 'Yacht'
]
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])
num_classes = len(class_name)

work_dir = 'work_dirs/gs3c_xview_rtmdetl_dota_init_20260619/train_epoch3'

load_from = (
    'weights/rotated_rtmdet_l-3x-dota_ms-2738da34_no_cls_dior_init.pth')
resume = False

max_epochs = 3
val_interval = 4
ckpt_interval = 1

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=ckpt_interval,
        save_last=True,
        max_keep_ckpts=4))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_interval)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_labeled/labelTxt/',
        data_prefix=dict(img_path='train_labeled/images/')))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/annfiles/',
        data_prefix=dict(img_path='test/images/'),
        filter_cfg=dict(filter_empty_gt=False)))

test_dataloader = val_dataloader

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(num_classes=num_classes))
