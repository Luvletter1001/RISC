_base_ = './hrrsd_rtmdet_l_dota_init_internal_epoch3_gpu0189.py'

work_dir = (
    'work_dirs/benchmark_p15l_vs_openrsd_20260624/'
    'openrsd_rtmdet_l_overfit100_b8')

load_from = None
resume = False
custom_hooks = []

# Match the P15 overfit100 benchmark data口径: HRRSD train split, first 100
# images, 800x800 preprocessing, batch size 8. This config is test-only.
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(indices=100))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        indices=100,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=False)))

test_dataloader = val_dataloader
