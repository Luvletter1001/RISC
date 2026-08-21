_base_ = './hrrsd_p15d_logit_fused_gaussian_quality_overfit100_4gpu_b8_v5.py'

work_dir = 'work_dirs/p15f_aux_only_gaussian_quality_hrrsd_20260623/overfit100_4gpu_b8_v5'

# P15F keeps Gaussian quality as auxiliary supervision but removes it from
# inference-time posterior scoring by setting alpha to zero.
model = dict(
    bbox_head=dict(
        quality_logit_alpha=0.0,
        learnable_quality_logit_alpha=False))

train_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
val_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
auto_scale_lr = dict(enable=False, base_batch_size=32)
