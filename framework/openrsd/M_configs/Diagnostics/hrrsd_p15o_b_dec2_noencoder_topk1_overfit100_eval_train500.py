_base_ = './hrrsd_p15o_b_dec2_noencoder_topk1_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p17_strict_e2e_subset_parity_hrrsd_20260626/'
    'diag_p15o_b_overfit100_best_eval_train500')

# Diagnostic only: evaluate the existing P15O-B overfit100 checkpoint on the
# first 500 train images. This separates overfit100 memorization from larger
# train-subset capacity.
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        indices=500,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/')))
test_dataloader = val_dataloader
