_base_ = './hrrsd_p16t_p15o_b_dec2_resnet50init_objneg_q200_dupneg020_latedecay_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p18_strict_e2e_scaleup_hrrsd_20260626/'
    'p18b_p16t_scratch_train1000_gpu2389_b32_v5')

# Backup for P18A: same strict no-head/no-NMS P16T recipe, but train the
# train[:1000] subset from the normal ImageNet backbone initialization instead
# of curriculum-loading the train500 checkpoint.
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

auto_scale_lr = dict(enable=False, base_batch_size=128)

randomness = dict(seed=3407, deterministic=False)
