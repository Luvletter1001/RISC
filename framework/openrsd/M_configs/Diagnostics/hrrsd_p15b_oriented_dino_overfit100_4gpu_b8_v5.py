_base_ = './hrrsd_p15b_oriented_dino_overfit100_4gpu_b4_v5.py'

work_dir = 'work_dirs/p15b_scod_hrrsd_20260623/overfit100_4gpu_b8_v5'

train_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
val_dataloader = dict(batch_size=8, num_workers=4, dataset=dict(indices=100))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=32)
