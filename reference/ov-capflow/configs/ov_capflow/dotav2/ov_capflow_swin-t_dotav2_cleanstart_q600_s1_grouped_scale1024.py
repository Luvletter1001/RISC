_base_ = ['./ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped.py']

physical_gpus = (8, 9)
selected_world_size = 2
matched_control = '8-S1-G-R-seed20260716-scale800'

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

model = dict(backbone=dict(with_cp=False))
model_wrapper_cfg = dict(static_graph=False)
train_dataloader = dict(
    batch_size=2,
    batch_sampler=dict(
        update_count_multiple=2,
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            's1_grouped_scale1024_seed20260716_gpu89_sampler.json')),
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader

optim_wrapper = dict(accumulative_counts=2)
randomness = dict(seed=20260716, deterministic=False, diff_rank_seed=False)
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    's1_grouped_scale1024_seed20260716_gpu89')
resume = False
