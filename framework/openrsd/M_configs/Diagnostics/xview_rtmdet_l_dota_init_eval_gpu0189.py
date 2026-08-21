_base_ = './xview_rtmdet_l_dota_init_epoch3_gpu0189.py'

work_dir = 'work_dirs/gs3c_xview_rtmdetl_dota_init_20260619/eval_baseline_full'

load_from = None
resume = False

test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root='/data1/zcy/datasets/xView_New_800_600/',
        ann_file='test/annfiles/',
        data_prefix=dict(img_path='test/images/'),
        filter_cfg=dict(filter_empty_gt=False)))

val_dataloader = test_dataloader
