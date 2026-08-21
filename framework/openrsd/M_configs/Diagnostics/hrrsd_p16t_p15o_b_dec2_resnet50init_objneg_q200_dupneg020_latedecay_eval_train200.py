_base_ = './hrrsd_p16t_p15o_b_dec2_resnet50init_objneg_q200_dupneg020_latedecay_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p17_strict_e2e_subset_parity_hrrsd_20260626/'
    'diag_p16t_best_eval_train200')

# Diagnostic only: evaluate P16T's best checkpoint on train[:200] using the
# OpenRSD baseline-subset metric. The base model remains strict no-head/no-NMS.
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        indices=200,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/')))
test_dataloader = val_dataloader
