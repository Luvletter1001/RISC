_base_ = './hrrsd_p15b_oriented_dino_overfit100_4gpu_b8_v5.py'

work_dir = 'work_dirs/p15h_objectness_tempered_hrrsd_20260623/overfit100_4gpu_b8_v5'

# P15H-A keeps the P15B final posterior unchanged, but weakens the
# objectness gate used by encoder top-k query selection and Hungarian matching.
model = dict(
    bbox_head=dict(
        topk_objectness_power=0.5,
        matching_objectness_power=0.5,
        objectness_prior_floor=0.05))

train_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
val_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=32)
