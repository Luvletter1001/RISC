_base_ = './ov_capflow_swin-t_hrsc_c1_fusion_5e.py'

model = dict(
    bbox_head=dict(
        balanced_cfg=dict(
            enabled=True, matched_weight=1.0, unmatched_weight=1.0)))
work_dir = 'work_dirs/ov_capflow_hrsc/hrsc_c2_balanced_5e'
