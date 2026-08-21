_base_ = './hrrsd_p15b_oriented_dino_overfit100_4gpu_b8_v5.py'

work_dir = 'work_dirs/p15i_objectness_gated_support_hrrsd_20260623/overfit100_4gpu_b8_v5'

# P15I-A keeps P15B top-k, matching, and final posterior unchanged. It only
# gates support-conditioned query initialization by encoder objectness.
model = dict(
    bbox_head=dict(
        support_objectness_gate_power=0.5))

train_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
val_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=32)
