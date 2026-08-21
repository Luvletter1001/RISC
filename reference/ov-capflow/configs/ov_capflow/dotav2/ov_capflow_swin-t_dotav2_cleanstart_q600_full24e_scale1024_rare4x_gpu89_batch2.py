_base_ = ['./ov_capflow_swin-t_dotav2_cleanstart_q600_full24e.py']

# T6 is the first scale-1024 screen to pass the frozen AP50, novel4 and base14
# endpoint gates. The full-data promotion starts again from the generic
# GroundingDINO-compatible checkpoint; it never imports the S1 checkpoint.
promoted_screen_winner = '8-T6-R-E12'
promoted_scientific_deltas = (
    'scale_800_to_1024', 'fixed_size_rare_positive_exposure_4x')
physical_gpus = (8, 9)
selected_world_size = 2

rare_repeat_factor = 4
rare_unique_train_images = 824
rare_train_exposures = 3296
replaced_empty_images = 2472
full_train_empty_images = 20103
rare_full_root = (
    'work_dirs/dotav2_cleanstart/full_rare4x_seed20260718/train/')

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities')),
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities')),
]

# Two exclusive A40s keep the full24 effective batch of 32. A real worst-batch
# preflight must pass before launch; a resource-only fallback may use
# batch1/GPU x accumulation16 without changing the scientific recipe.
train_dataloader = dict(
    batch_size=2,
    batch_sampler=dict(
        update_count_multiple=8,
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            'full24_scale1024_rare4x_seed20260716_gpu89_b2_epoch.json')),
    dataset=dict(
        data_root=rare_full_root,
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        data_root='/data1/zcy/datasets/DOTA2_1024_500/',
        ann_file='ss_val/annfiles/',
        data_prefix=dict(img_path='ss_val/images/'),
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

optim_wrapper = dict(accumulative_counts=8)
randomness = dict(seed=20260716, deterministic=False, diff_rank_seed=False)
resume = False
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    'full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2')
