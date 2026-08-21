_base_ = './hrsc2016_ship1_rtmdet_tiny_gs3c_g1_light_gpu0_eval.py'

test_dataloader = dict(
    batch_size=16,
    num_workers=0,
    persistent_workers=False)

val_dataloader = test_dataloader

work_dir = (
    'work_dirs/gs3c_hrsc_ship1_scope_20260619/'
    'hrsc_gpu0_rtmdet_tiny_gs3c_g1_light_eval_noworkers')
