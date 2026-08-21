_base_ = './shiprs_rtmdet_l_dota_init_eval_gpu0189.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.gs3c_rtmdet_head'],
    allow_failed_imports=False)

work_dir = 'work_dirs/gs3c_shiprs_rtmdetl_dota_init_20260619/eval_g1_light_full'

shiprs_class_names = [
    'aoe', 'arleigh_burke_dd', 'asagiri_dd', 'atago_dd',
    'austin_ll', 'barge', 'cargo', 'commander', 'container_ship', 'dock',
    'enterprise', 'epf', 'ferry', 'fishing_vessel', 'hatsuyuki_dd',
    'hovercraft', 'hyuga_dd', 'lha_ll', 'lsd_41_ll', 'masyuu_as',
    'medical_ship', 'midway', 'motorboat', 'nimitz', 'oil_tanker',
    'osumi_ll', 'other_aircraft_carrier', 'other_auxiliary_ship',
    'other_destroyer', 'other_frigate', 'other_landing', 'other_merchant',
    'other_ship', 'other_warship', 'patrol', 'perry_ff', 'roro',
    'sailboat', 'sanantonio_as', 'submarine', 'test_ship', 'ticonderoga',
    'training_ship', 'tugboat', 'wasp_ll', 'yacht', 'yudao_ll',
    'yudeng_ll', 'yuting_ll', 'yuzhao_ll'
]

model = dict(
    bbox_head=dict(
        type='GSRRotatedRTMDetSepBNHead',
        gaussian_semantic_scale=dict(
            enable=True,
            class_area_priors_csv=(
                'work_dirs/gs3c_dataset_inventory_20260619/'
                'shiprs_train_split_area_priors.csv'),
            class_names=shiprs_class_names,
            mode='continuous_logit_energy',
            z0=5.0,
            beta=0.34657359028,
            adapter_hidden=64,
            adapter_nonpositive_delta=True,
            preserve_s3c_guard=True,
            guard_lambda=0.25,
            domain_mode='closed_set')))
