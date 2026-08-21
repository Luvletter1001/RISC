_base_ = './hrrsd_p16t_p15o_b_dec2_resnet50init_objneg_q200_dupneg020_latedecay_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p18_strict_e2e_scaleup_hrrsd_20260626/'
    'p18a_p16t_curriculum_train1000_gpu2389_b32_v5')

# P18A scale-up step:
# P16T already proves strict no-head/no-NMS parity on train[:500]. Continue
# from its best checkpoint and enlarge the overfit-subset target to train[:1000].
load_from = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16t_p15o_b_dec2_resnet50init_objneg_q200_dupneg020_latedecay_generalization500_gpu89_b32_v5/'
    'best_dota_mAP_epoch_60.pth')

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(indices=1000))

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        indices=1000,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/')))
test_dataloader = val_dataloader

# Four GPUs x 32 images/GPU. Auto scaling stays disabled; this records the
# actual global batch used for audit.
auto_scale_lr = dict(enable=False, base_batch_size=128)

randomness = dict(seed=3407, deterministic=False)
