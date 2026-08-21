_base_ = './dota2_p23a_strict_e2e_full5e_from_p22b_b16x2_v1.py'

# P23C: conservative full-DOTA2 recovery run after P23A b16/GPU and P23B
# b8/GPU both OOMed on dense full-data batches. Keep the same strict E2E
# continuation setup and lower per-GPU batch to 4.
work_dir = (
    'work_dirs/p23_strict_e2e_dota2_full_20260626/'
    'p23c_full5e_from_p22b_b4x2_v1')

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(indices=None))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(indices=None))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=8)
