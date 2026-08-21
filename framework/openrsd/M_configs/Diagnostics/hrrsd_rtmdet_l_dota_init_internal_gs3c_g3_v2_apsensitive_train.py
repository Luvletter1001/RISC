_base_ = './hrrsd_rtmdet_l_dota_init_internal_gs3c_g3_poshardneg_train_gpu0189.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_g3_v2_apsensitive_seed3407_e2')

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
            consistency_loss=dict(
                enable=True,
                loss_weight=0.05,
                margin=0.5,
                min_logprob_gap=1.0,
                max_hardneg=1,
                min_hardneg_logit=0.0,
                max_gt_abs_z=3.0,
                log_stats=True))))
