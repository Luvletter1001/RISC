_base_ = './hrrsd_rtmdet_l_dota_init_internal_epoch3_gpu0189.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'diag_openrsd_epoch3_eval_val500')

load_from = None
resume = False
custom_hooks = []

# Diagnostic only: evaluate the existing OpenRSD epoch-3 HRRSD checkpoint on
# the same val[:500] protocol used by P16A. This validates whether the val500
# split and evaluator are capable of producing non-trivial mAP.
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        indices=500,
        ann_file='val/labelTxt/',
        data_prefix=dict(img_path='val/images/'),
        filter_cfg=dict(filter_empty_gt=False)))
test_dataloader = val_dataloader
