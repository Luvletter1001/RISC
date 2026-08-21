_base_ = './hrrsd_p19a_p18a_curriculum_train2000_gpu2389_b32_v5.py'

work_dir = (
    'work_dirs/p20_strict_e2e_scaleup_hrrsd_20260626/'
    'p20a_p19a_curriculum_train7000_gpu2389_b32_v5')

# P20A scale-up step:
# P19A proves strict no-head/no-NMS parity on train[:2000]. Continue from its
# epoch-10 best checkpoint and enlarge the target to near-full train[:7000].
load_from = (
    'work_dirs/p19_strict_e2e_scaleup_hrrsd_20260626/'
    'p19a_p18a_curriculum_train2000_gpu2389_b32_v5/'
    'best_dota_mAP_epoch_10.pth')

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(indices=7000))

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        indices=7000,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/')))
test_dataloader = val_dataloader

# Four GPUs x 32 images/GPU. Auto scaling stays disabled; this records the
# actual global batch used for audit.
auto_scale_lr = dict(enable=False, base_batch_size=128)

randomness = dict(seed=3407, deterministic=False)
