_base_ = './hrrsd_p16t_p15o_b_dec2_resnet50init_objneg_q200_dupneg020_latedecay_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p17_strict_e2e_subset_parity_hrrsd_20260626/'
    'diag_p16t_best_eval_train1000')

# Larger-subset diagnostic for the current strict no-head/no-NMS mainline.
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        indices=1000,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/')))
test_dataloader = val_dataloader
