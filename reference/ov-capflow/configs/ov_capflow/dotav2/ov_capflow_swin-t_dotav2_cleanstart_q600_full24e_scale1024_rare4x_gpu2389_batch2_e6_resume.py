_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_'
    'full24e_scale1024_rare4x_gpu89_batch2.py'
]

# Continue the exact T7 scientific recipe after the completed Epoch 6 raw
# validation. Four A40s halve the microsteps per epoch while preserving the
# effective batch and optimizer-update count of the original two-GPU run.
continuation_from_epoch = 6
physical_gpus = (2, 3, 8, 9)
selected_world_size = 4

train_dataloader = dict(
    batch_size=2,
    batch_sampler=dict(
        update_count_multiple=4,
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            'full24_scale1024_rare4x_seed20260716_'
            'gpu2389_b2_e6_resume_epoch.json')))
optim_wrapper = dict(accumulative_counts=4)

work_dir = (
    'work_dirs/dotav2_cleanstart/'
    'full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2')
load_from = work_dir + '/epoch_6_world4_iterrebased.pth'
resume = True
