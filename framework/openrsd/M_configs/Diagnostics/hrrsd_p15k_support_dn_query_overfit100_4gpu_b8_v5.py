_base_ = './hrrsd_p15b_oriented_dino_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15k_support_dn_query_hrrsd_20260624/'
    'overfit100_4gpu_b8_v5_a')

# P15K-A is a training-only support-conditioned denoising experiment.
# Inference remains the P15B strict-E2E path: fixed query set, no dense head,
# no NMS, and no score-threshold fallback.
model = dict(
    bbox_head=dict(
        support_dn_query_scale=0.5,
        support_dn_query_logit=6.0,
        support_dn_query_warmup_iters=80))

train_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
val_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=32)
