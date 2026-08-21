_base_ = './hrrsd_p15o_b_dec2_noencoder_topk1_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16a_p15o_b_dec2_noencoder_topk1_generalization500_gpu89_b8_v5')

# P16A keeps the current strongest strict-E2E P15O-B model path and changes
# only the data protocol: train on HRRSD train[:500], evaluate on val[:500].
# The base P15 overfit configs evaluate on train/, so val paths are explicit
# here to make this a real generalization probe instead of another overfit run.
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(indices=500))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        indices=500,
        ann_file='val/labelTxt/',
        data_prefix=dict(img_path='val/images/')))
test_dataloader = val_dataloader

# Two GPUs x 8 images/GPU. Auto scaling is disabled, but keep the recorded
# reference batch size accurate for later audit.
auto_scale_lr = dict(enable=False, base_batch_size=16)

randomness = dict(seed=3407, deterministic=False)
