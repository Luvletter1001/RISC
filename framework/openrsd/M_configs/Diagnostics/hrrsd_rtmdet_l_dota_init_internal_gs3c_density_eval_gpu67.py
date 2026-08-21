_base_ = './hrrsd_rtmdet_l_dota_init_internal_eval_gpu0189.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.gs3c_rtmdet_head'],
    allow_failed_imports=False)

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'eval_epoch3_plus1_density_w005_full')

load_from = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus1_density_w005/epoch_1.pth')
resume = False

hrrsd_class_names = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]

model = dict(
    bbox_head=dict(
        type='GSRRotatedRTMDetSepBNHead',
        gaussian_semantic_scale=dict(
            enable=True,
            class_area_priors_csv=(
                'work_dirs/gs3c_dataset_inventory_20260619/'
                'hrrsd_internal_train_area_priors.csv'),
            class_names=hrrsd_class_names,
            mode='continuous_logit_energy',
            z0=4.0,
            beta=0.0,
            domain_mode='closed_set',
            consistency_loss=dict(enable=False),
            density_head=dict(
                enable=True,
                loss_weight=0.05,
                hidden_channels=64,
                max_delta_abs=0.25,
                nonpositive_delta=True,
                log_stats=False))))
