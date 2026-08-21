_base_ = './hrrsd_p15f_aux_only_gaussian_quality_overfit100_4gpu_b8_v5.py'

work_dir = 'work_dirs/p15g_weak_aux_gaussian_quality_hrrsd_20260623/overfit100_4gpu_b8_v5'

# P15G-A keeps the P15B-mainline inference posterior unchanged while reducing
# the training-only Gaussian quality regularizer that hurt P15F.
model = dict(
    bbox_head=dict(
        loss_quality_weight=0.25,
        quality_logit_alpha=0.0,
        learnable_quality_logit_alpha=False))

train_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
val_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=32)
