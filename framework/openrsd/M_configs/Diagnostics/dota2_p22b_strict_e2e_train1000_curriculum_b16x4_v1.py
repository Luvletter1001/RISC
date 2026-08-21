_base_ = './dota2_p22a_strict_e2e_train1000_curriculum_b32x4_v1.py'

# DOTA2 has much denser images than HRRSD. P22A reached OOM at b32/GPU on
# train[:1000], so keep the strict E2E design unchanged and lower only the
# per-GPU batch size to finish the 1000-image comparison cleanly.
work_dir = (
    'work_dirs/p22_strict_e2e_dota2_subset_20260626/'
    'p22b_train1000_curriculum_from_p21a_b16x4_v1')

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(indices=1000))

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(indices=1000))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=64)
