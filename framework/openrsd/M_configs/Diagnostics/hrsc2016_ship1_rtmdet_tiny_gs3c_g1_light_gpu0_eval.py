_base_ = './hrsc2016_ship1_rtmdet_tiny_gpu_eval.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.gs3c_rtmdet_head'],
    allow_failed_imports=False)

_gs3c_class_names = ['ship']

model = dict(
    bbox_head=dict(
        type='GSRRotatedRTMDetSepBNHead',
        gaussian_semantic_scale=dict(
            enable=True,
            class_area_priors_csv=(
                'work_dirs/gs3c_dataset_inventory_20260619/'
                'hrsc2016_ship1_train_area_priors.csv'),
            class_names=_gs3c_class_names,
            domain_mode='closed_set',
            mode='continuous_logit_energy',
            z0=5.0,
            beta=0.34657359028,
            adapter_hidden=64,
            adapter_nonpositive_delta=True)))

work_dir = (
    'work_dirs/gs3c_hrsc_ship1_scope_20260619/'
    'hrsc_gpu0_rtmdet_tiny_gs3c_g1_light_eval')
