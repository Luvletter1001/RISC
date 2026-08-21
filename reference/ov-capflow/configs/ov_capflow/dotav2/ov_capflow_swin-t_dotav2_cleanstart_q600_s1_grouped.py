_base_ = ['./ov_capflow_swin-t_dotav2_cleanstart_q600_s1_control.py']

model = dict(
    train_query_groups=3,
    bbox_head=dict(matching_query_groups=3))
train_dataloader = dict(
    batch_sampler=dict(num_matching_queries=1800))
work_dir = 'work_dirs/dotav2_cleanstart/s1_grouped'
