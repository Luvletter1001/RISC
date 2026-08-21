_base_ = './whu_mix_rtmdet_l_dota_init_internal_eval_gpu2567.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.gs3c_rtmdet_head'],
    allow_failed_imports=False)

work_dir = (
    'work_dirs/gs3c_whu_mix_rtmdetl_dota_init_20260619/'
    'eval_epoch3_resume_g1_z4_full')

model = dict(
    bbox_head=dict(
        type='GSRRotatedRTMDetSepBNHead',
        gaussian_semantic_scale=dict(
            enable=True,
            class_area_priors_csv=(
                'work_dirs/gs3c_priors/'
                'whu_mix_internal_split_20260619_train_class_area_priors.csv'),
            class_names=['building'],
            mode='continuous_logit_energy',
            z0=4.0,
            beta=1.38629436112,
            adapter_hidden=64,
            adapter_nonpositive_delta=True,
            preserve_s3c_guard=True,
            guard_lambda=0.25,
            domain_mode='closed_set')))

