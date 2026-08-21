_base_ = './hrrsd_p15b_oriented_dino_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15l_class_balanced_query_hrrsd_20260624/'
    'overfit100_4gpu_b8_v5_a')

# P15L-A changes training-time query selection only. Inference still uses the
# P15B global top-k query set and final strict-E2E posterior.
model = dict(
    bbox_head=dict(
        train_class_balanced_topk_per_class=2))

train_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
val_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=32)
