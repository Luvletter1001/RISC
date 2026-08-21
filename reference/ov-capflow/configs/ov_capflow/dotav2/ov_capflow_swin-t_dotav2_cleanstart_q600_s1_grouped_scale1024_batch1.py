_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_scale1024.py'
]

recovery_of = '8-T2-R-magma-shared-memory-abort'
train_dataloader = dict(
    batch_size=1,
    batch_sampler=dict(
        update_count_multiple=4,
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            's1_grouped_scale1024_seed20260716_gpu89_batch1_sampler.json')))
optim_wrapper = dict(accumulative_counts=4)
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    's1_grouped_scale1024_seed20260716_gpu89_batch1')
resume = False
