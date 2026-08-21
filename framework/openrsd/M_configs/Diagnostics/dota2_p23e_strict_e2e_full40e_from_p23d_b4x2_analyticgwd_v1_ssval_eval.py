_base_ = './dota2_p23e_strict_e2e_full40e_from_p23d_b4x2_analyticgwd_v1.py'

# Evaluation-only wrapper for paper-style DOTA2 val-set checking.
# The training config keeps its internal validation split unchanged.
work_dir = (
    'work_dirs/p23_strict_e2e_dota2_full_20260627/'
    'p23e_full40e_from_p23d_epoch5_b4x2_analyticgwd_v1_ssval_eval')

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
