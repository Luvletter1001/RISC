_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_d12_control.py'
]

physical_gpus = (0, 1, 2, 3, 4)
d12_role = 'candidate'
d12_only_scientific_delta = (
    'initialize_decoder_branches_0_5_terminal_xywh_from_raw_ogc')

train_dataloader = dict(
    batch_sampler=dict(
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            'd12_world5_candidate_seed20260716_gpu01234_sampler.json')))

load_from = (
    'work_dirs/dotav2_cleanstart/checkpoints/'
    'groundingdino_swint_ogc_q600_d12_w5_xywh_seed20260716.pth')
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    'd12_world5_candidate_seed20260716_gpu01234_batch2')
resume = False
