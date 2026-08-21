_base_ = '../OfficialMMRotateWeightRepro/redet/redet-le90_re50_refpn_1x_dota.py'

model = dict(backbone=dict(init_cfg=None))

data_root = '/data1/zcy/OpenRSD/data/DOTA1_1024_500/'

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='ss_val/annfiles/',
        data_prefix=dict(img_path='ss_val/images/')))

val_dataloader = test_dataloader

work_dir = '/data1/zcy/OpenRSD/work_dirs/rotation_study_36h/dota1_redet_eval'
