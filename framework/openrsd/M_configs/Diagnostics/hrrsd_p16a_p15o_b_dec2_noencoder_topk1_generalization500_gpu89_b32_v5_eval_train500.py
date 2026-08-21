_base_ = './hrrsd_p16a_p15o_b_dec2_noencoder_topk1_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'diag_p16a_b32_best_eval_train500')

# Diagnostic only: evaluate the same P16A checkpoint on train[:500]. If this
# is also low, the model did not even learn the train split. If this is high
# while val[:500] is low, the failure is train/val generalization.
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        indices=500,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/')))
test_dataloader = val_dataloader
