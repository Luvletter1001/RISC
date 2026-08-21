_base_ = './hrrsd_p15b_oriented_dino_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15j_support_gate_warmup_floor_hrrsd_20260624/'
    'overfit100_4gpu_b8_v5_b2_warmup200')

# P15J-B2 tests whether introducing the objectness gate slightly earlier
# improves P15J-A's e55/e80 stability while keeping the same floor schedule.
model = dict(
    bbox_head=dict(
        support_objectness_gate_power=0.5,
        support_objectness_gate_warmup_iters=200,
        support_objectness_gate_floor_start=0.5,
        support_objectness_gate_floor_end=0.2))

train_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
val_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=32)
