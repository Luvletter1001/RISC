_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x.py'
]

physical_gpus = (5, 6, 7, 8, 9)
selected_world_size = 5
d12_pair_id = 'D12-Q600-terminal-xywh-world5-seed20260716'
d12_role = 'control'
d12_only_scientific_delta = 'none'

train_dataloader = dict(
    batch_size=2,
    batch_sampler=dict(
        update_count_multiple=1,
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            'd12_world5_control_seed20260716_gpu56789_sampler.json')))
optim_wrapper = dict(accumulative_counts=1)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=-1,
        save_best=None,
        rule=None,
        save_last=True))

load_from = (
    'work_dirs/dotav2_cleanstart/checkpoints/'
    'groundingdino_swint_ogc_q600_d12_w5_control_seed20260716.pth')
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    'd12_world5_control_seed20260716_gpu56789_batch2')
resume = False
