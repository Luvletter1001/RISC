_base_ = './hrrsd_p16a_p15o_b_dec2_noencoder_topk1_generalization500_gpu89_b8_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16a_p15o_b_dec2_noencoder_topk1_generalization500_gpu89_b32_v5')

# 4x the previous P16A launch: 32 images/GPU on two GPUs, total batch 64.
train_dataloader = dict(batch_size=32, num_workers=4)
val_dataloader = dict(batch_size=32, num_workers=4)
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=64)
