_base_ = '../OfficialMMRotateWeightRepro/rtmdet/rotated_rtmdet_tiny_hrsc_dota_eval.py'

data_root = '/data1/zcy/datasets/HRSC2016_DOTA_ship1/'

test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file='test/labelTxt/',
        data_prefix=dict(img_path='test/images/')))

val_dataloader = test_dataloader
