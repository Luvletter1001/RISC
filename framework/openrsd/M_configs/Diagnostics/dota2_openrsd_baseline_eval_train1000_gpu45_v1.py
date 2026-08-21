_base_ = './dota2_openrsd_baseline_eval_train500_gpu45_v1.py'

work_dir = (
    'work_dirs/p22_strict_e2e_dota2_subset_20260626/'
    'baseline_openrsd_eval_train1000_gpu45_v1')

val_dataloader = dict(dataset=dict(indices=1000))
test_dataloader = val_dataloader
