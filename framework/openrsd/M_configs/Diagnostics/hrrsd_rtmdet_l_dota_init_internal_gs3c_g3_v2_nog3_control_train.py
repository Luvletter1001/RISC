_base_ = './hrrsd_rtmdet_l_dota_init_internal_gs3c_g3_poshardneg_train_gpu0189.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_g3_v2_nog3_ctrl_seed3407_e2')

max_epochs = 2
val_interval = 3

randomness = dict(seed=3407, deterministic=False)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_interval)

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            enable=False,
            beta=0.0,
            consistency_loss=dict(
                enable=False,
                loss_weight=0.0,
                log_stats=False))))
