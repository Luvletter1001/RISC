_base_ = (
    '../../work_dirs/gs3c_whu_mix_rtmdetl_dota_init_20260619/'
    'train_epoch3/G02_Baselines_Data11_WHU_Mix_M10_RTMDet_L.py')

work_dir = (
    'work_dirs/gs3c_whu_mix_rtmdetl_dota_init_20260619/'
    'eval_epoch3_resume_baseline_full')

load_from = None
resume = False

test_dataloader = dict(batch_size=8, num_workers=2, persistent_workers=True)
val_dataloader = test_dataloader

