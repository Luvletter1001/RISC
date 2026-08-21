_base_ = ['./ov_capflow_swin-t_dotav2_cleanstart_q600_base.py']

subset_root = (
    '/data1/zcy/OV-CapFlow/work_dirs/dotav2_cleanstart/'
    'subsets/seed20260715/')
subset_manifest_sha256 = (
    'a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e')

train_dataloader = dict(
    batch_size=3,
    dataset=dict(
        data_root=subset_root,
        ann_file='s0/annfiles/',
        data_prefix=dict(img_path='s0/images/'),
        filter_cfg=dict(filter_empty_gt=False)))
val_dataloader = dict(
    dataset=dict(
        data_root=subset_root,
        ann_file='s0/annfiles/',
        data_prefix=dict(img_path='s0/images/'),
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=True))
test_dataloader = val_dataloader

train_cfg = dict(
    _delete_=True, type='IterBasedTrainLoop', max_iters=50,
    val_interval=50)
param_scheduler = []
optim_wrapper = dict(accumulative_counts=1)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=5),
    checkpoint=dict(
        by_epoch=False, interval=50, max_keep_ckpts=1,
        save_last=True))
log_processor = dict(by_epoch=False)
work_dir = 'work_dirs/dotav2_cleanstart/s0_control'
resume = False
