_base_ = './dota2_p23f_strict_e2e_full80e_from_p23e_b4x4_arborlr_v1.py'

# Evaluation-only wrapper for paper-style DOTA2 val-set checking.
# The training config intentionally keeps its internal ss_train validation;
# this wrapper switches only test/val dataloaders to ss_val.
work_dir = (
    'work_dirs/p23_strict_e2e_dota2_full_20260628/'
    'p23f_full80e_from_p23e_epoch40_b4x4_arborlr_v1_ssval_eval')

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        ann_file='ss_val/annfiles/',
        data_prefix=dict(img_path='ss_val/images/'),
        indices=None))
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator
