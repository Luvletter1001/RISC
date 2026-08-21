_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_'
    'full24e_scale1024_rare4x_gpu89_batch2.py'
]

# Resource-only contingency for a production OOM in 8-T7-F. Keep the model,
# data exposure, batch size, accumulation and optimization protocol unchanged;
# split more high-query-area global updates into singleton local batches.
resource_fallback_of = '8-T7-F'
train_dataloader = dict(
    batch_sampler=dict(
        max_query_area=20_000_000,
        update_count_multiple=8,
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            'full24_scale1024_rare4x_seed20260716_gpu89_b2_'
            'maxqa20m_epoch.json')))
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    'full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2_maxqa20m')
resume = False
