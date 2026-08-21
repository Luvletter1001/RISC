_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world3_v2_control.py'
]

physical_gpus = (0, 1, 2)
d11_v2_role = 'candidate'
d11_v2_only_scientific_delta = (
    'initialize_q600_content_queries_from_raw_ogc_first600')

train_dataloader = dict(
    batch_sampler=dict(
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            'd11_v2_world3_candidate_seed20260716_gpu012_sampler.json')))

load_from = (
    'work_dirs/dotav2_cleanstart/checkpoints/'
    'groundingdino_swint_ogc_q600_d11_content_first600_v2.pth')
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    'd11_v2_world3_candidate_seed20260716_gpu012_batch2')
resume = False
