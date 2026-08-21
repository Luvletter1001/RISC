_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x.py'
]
a1_parent = '8-T6-R-E12'
a1_only_scientific_delta = (
    'rotated_iou_positive_classification_target')
model = dict(
    bbox_head=dict(position_supervised_cfg=dict(enabled=True)))
train_dataloader = dict(
    batch_sampler=dict(
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            's1_grouped_scale1024_seed20260716_gpu89_batch1_rare4x_'
            'position_supervised_sampler.json')))
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    's1_grouped_scale1024_seed20260716_gpu89_batch1_rare4x_'
    'position_supervised')
resume = False
