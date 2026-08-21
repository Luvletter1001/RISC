_base_ = './dota2_p23a_strict_e2e_full5e_from_p22b_b16x2_v1.py'

# P23B: recovery run after P23A b16/GPU OOMed on full DOTA2 at epoch 1.
# Keep the same full-data 5E continuation setup, but lower per-GPU batch to 8.
work_dir = (
    'work_dirs/p23_strict_e2e_dota2_full_20260626/'
    'p23b_full5e_from_p22b_b8x2_v1')

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(indices=None))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(indices=None))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=16)
