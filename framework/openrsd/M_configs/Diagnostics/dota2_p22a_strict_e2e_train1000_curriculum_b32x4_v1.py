_base_ = './dota2_p21a_strict_e2e_train500_resnet50_b32x4_v1.py'

work_dir = (
    'work_dirs/p22_strict_e2e_dota2_subset_20260626/'
    'p22a_train1000_curriculum_from_p21a_b32x4_v1')

load_from = (
    'work_dirs/p21_strict_e2e_dota2_subset_20260626/'
    'p21a_train500_resnet50_b32x4_v1/'
    'p21_best_for_p22.pth')

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(indices=1000))

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(indices=1000))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=128)
randomness = dict(seed=3407, deterministic=False)
