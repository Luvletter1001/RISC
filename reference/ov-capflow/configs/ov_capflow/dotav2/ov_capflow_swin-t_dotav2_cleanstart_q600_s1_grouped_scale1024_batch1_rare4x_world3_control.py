_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x.py'
]

physical_gpus = (3, 4, 5)
selected_world_size = 3
d11_pair_id = 'D11-Q600-content-first600-world3-seed20260716'
d11_role = 'control'
d11_only_scientific_delta = 'none'

train_dataloader = dict(
    batch_size=1,
    batch_sampler=dict(
        update_count_multiple=3,
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            'd11_world3_control_seed20260716_gpu345_sampler.json')))
optim_wrapper = dict(accumulative_counts=3)

load_from = (
    'work_dirs/dotav2_cleanstart/checkpoints/'
    'groundingdino_swint_ogc_q600_compatible.pth')
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    'd11_world3_control_seed20260716_gpu345_batch1')
resume = False
