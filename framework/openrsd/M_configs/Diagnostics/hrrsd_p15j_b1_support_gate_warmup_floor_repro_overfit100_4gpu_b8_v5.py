_base_ = './hrrsd_p15b_oriented_dino_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15j_support_gate_warmup_floor_hrrsd_20260624/'
    'overfit100_4gpu_b8_v5_b1_repro')

# P15J-B1 is an exact-method reproducibility run for P15J-A.  It keeps
# P15B top-k, Hungarian matching, and final posterior unchanged, and repeats
# the same support-query warmup/floor gate that reached best mAP=0.6266.
model = dict(
    bbox_head=dict(
        support_objectness_gate_power=0.5,
        support_objectness_gate_warmup_iters=240,
        support_objectness_gate_floor_start=0.5,
        support_objectness_gate_floor_end=0.2))

train_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
val_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=32)
