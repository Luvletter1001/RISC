_base_ = './hrrsd_p18a_p16t_curriculum_train1000_gpu2389_b32_v5.py'

work_dir = (
    'work_dirs/p19_strict_e2e_scaleup_hrrsd_20260626/'
    'p19a_p18a_curriculum_train2000_gpu2389_b32_v5')

# P19A scale-up step:
# P18A proves strict no-head/no-NMS parity on train[:1000]. Continue from the
# epoch-15 best checkpoint and enlarge the overfit-subset target to train[:2000].
load_from = (
    'work_dirs/p18_strict_e2e_scaleup_hrrsd_20260626/'
    'p18a_p16t_curriculum_train1000_gpu2389_b32_v5/'
    'best_dota_mAP_epoch_15.pth')

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(indices=2000))

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        indices=2000,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/')))
test_dataloader = val_dataloader

# Four GPUs x 32 images/GPU. Auto scaling stays disabled; this records the
# actual global batch used for audit.
auto_scale_lr = dict(enable=False, base_batch_size=128)

randomness = dict(seed=3407, deterministic=False)
