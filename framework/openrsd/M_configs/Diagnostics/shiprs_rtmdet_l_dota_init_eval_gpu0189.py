_base_ = './shiprs_rtmdet_l_dota_init_epoch12_gpu0189.py'

data_root = '/data1/zcy/datasets/ShipRSImageNet_DOTA_split_20260619/'
work_dir = 'work_dirs/gs3c_shiprs_rtmdetl_dota_init_20260619/eval_baseline_full'

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        ann_file='val/labelTxt/',
        data_prefix=dict(img_path='val/images/'),
        filter_cfg=dict(filter_empty_gt=False)))

val_dataloader = test_dataloader
