_base_ = '../OfficialMMRotateWeightRepro/oriented_reppoints/oriented-reppoints-qbox_r50_fpn_mstrain-40e_dota.py'

load_from = '/data1/zcy/OpenRSD/weights/oriented_reppoints_r50_fpn_40e_dota_ms_le135-bb0323fd_no_cls.pth'

find_unused_parameters = True

num_classes = 18
data_root = '/data1/zcy/OpenRSD/data/DOTA2_1024_500/'

model = dict(
    bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    dataset=dict(
        type='DOTAv2Dataset',
        data_root=data_root,
        ann_file='ss_train/annfiles/',
        data_prefix=dict(img_path='ss_train/images/')))

val_dataloader = dict(
    dataset=dict(
        type='DOTAv2Dataset',
        data_root=data_root,
        ann_file='ss_val/annfiles/',
        data_prefix=dict(img_path='ss_val/images/')))

test_dataloader = val_dataloader

work_dir = 'work_dirs/dotav2_official_adapters/oriented_reppoints_r50_fpn_fullinit'
