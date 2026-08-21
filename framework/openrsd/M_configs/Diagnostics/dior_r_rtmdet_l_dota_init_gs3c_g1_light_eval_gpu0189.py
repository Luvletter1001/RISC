_base_ = './dior_r_rtmdet_l_dota_init_eval_gpu0189.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.gs3c_rtmdet_head'],
    allow_failed_imports=False)

work_dir = 'work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/eval_g1_light_full'

model = dict(
    bbox_head=dict(
        type='GSRRotatedRTMDetSepBNHead',
        gaussian_semantic_scale=dict(
            enable=True,
            class_area_priors_csv=(
                'work_dirs/gs3c_dior_scope_20260619/'
                'dior_r_trainval_area_priors.csv'),
            class_names=[
                'airplane', 'airport', 'baseballfield', 'basketballcourt',
                'bridge', 'chimney', 'dam', 'Expressway-Service-area',
                'Expressway-toll-station', 'golffield',
                'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
                'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
                'windmill'
            ],
            mode='continuous_logit_energy',
            z0=5.0,
            beta=0.34657359028,
            adapter_hidden=64,
            adapter_nonpositive_delta=True,
            preserve_s3c_guard=True,
            guard_lambda=0.25,
            domain_mode='closed_set')))
