_base_ = './hrrsd_rtmdet_l_dota_init_internal_eval_gpu0189.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.gs3c_rtmdet_head'],
    allow_failed_imports=False)

work_dir = 'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/eval_g1_light_full'

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
            z0=5.0,
            beta=0.34657359028,
            adapter_hidden=64,
            adapter_nonpositive_delta=True,
            preserve_s3c_guard=True,
            guard_lambda=0.25,
            domain_mode='closed_set')))
