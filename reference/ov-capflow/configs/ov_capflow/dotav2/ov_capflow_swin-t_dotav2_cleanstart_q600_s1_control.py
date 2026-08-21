_base_ = ['./ov_capflow_swin-t_dotav2_cleanstart_q600_base.py']

subset_root = (
    '/data1/zcy/OV-CapFlow/work_dirs/dotav2_cleanstart/'
    'subsets/seed20260715/')
subset_manifest_sha256 = (
    'a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e')

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=subset_root,
        ann_file='s1_train_all18/annfiles/',
        data_prefix=dict(img_path='s1_train_all18/images/'),
        filter_cfg=dict(filter_empty_gt=False)))
val_dataloader = dict(
    dataset=dict(
        data_root=subset_root,
        ann_file='s1_val_all18/annfiles/',
        data_prefix=dict(img_path='s1_val_all18/images/'),
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=True))
test_dataloader = val_dataloader

train_cfg = dict(
    _delete_=True, type='VariableBatchEpochBasedTrainLoop', max_epochs=12,
    val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0 / 10, by_epoch=False,
        begin=0, end=500),
    dict(
        type='CosineAnnealingLR', by_epoch=True, begin=0, end=12,
        eta_min=1e-6),
]
optim_wrapper = dict(accumulative_counts=2)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        by_epoch=True, interval=1, max_keep_ckpts=2,
        save_best='dota/mAP', rule='greater', save_last=True))
log_processor = dict(by_epoch=True)
work_dir = 'work_dirs/dotav2_cleanstart/s1_control'
resume = False
