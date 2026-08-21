_base_ = '../OfficialMMRotateWeightRepro/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py'

load_from = '/data1/zcy/OpenRSD/weights/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0_no_fc_cls.pth'

num_classes = 18
data_root = '/data1/zcy/OpenRSD/data/DOTA2_1024_500/'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes)))

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

work_dir = 'work_dirs/dotav2_official_adapters/oriented_rcnn_r50_fpn_fullinit'
