_base_ = './dota2_p22b_strict_e2e_train1000_curriculum_b16x4_v1.py'

# P24A: two-GPU replay of the P22B DOTA2 train1000 strict-E2E baseline with
# Arbor's current optimizer setting made explicit. This is tonight's control
# for GPU4/5 runs.
work_dir = (
    'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/'
    'p24a_arbor_replay_b16x2_seed3407')

optim_wrapper = dict(
    optimizer=dict(lr=2e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2))

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(indices=1000))
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(indices=1000))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=32)
randomness = dict(seed=3407, deterministic=False)
