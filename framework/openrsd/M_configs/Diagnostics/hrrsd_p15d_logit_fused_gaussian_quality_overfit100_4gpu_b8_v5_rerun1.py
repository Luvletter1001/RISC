_base_ = './hrrsd_p15d_logit_fused_gaussian_quality_overfit100_4gpu_b8_v5.py'

work_dir = 'work_dirs/p15d_logit_fused_gaussian_quality_hrrsd_20260623/overfit100_4gpu_b8_v5_rerun1'

# Keep the rerun explicitly at 8 images/GPU on 4 GPUs.
train_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
val_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
auto_scale_lr = dict(enable=False, base_batch_size=32)
