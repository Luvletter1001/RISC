_base_ = './hrrsd_p15b_oriented_dino_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15j_support_gate_warmup_floor_hrrsd_20260624/'
    'overfit100_4gpu_b8_v5_b3_floor060_025')

# P15J-B3 keeps P15J-A's warmup length but preserves more support delta
# through a higher residual floor.  This targets the weak ship/vehicle AP
# without touching final posterior, top-k selection, matching, NMS, or heads.
model = dict(
    bbox_head=dict(
        support_objectness_gate_power=0.5,
        support_objectness_gate_warmup_iters=240,
        support_objectness_gate_floor_start=0.6,
        support_objectness_gate_floor_end=0.25))

train_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
val_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=32)
