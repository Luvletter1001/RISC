_base_ = './p4_s3c_pre_nms_z4_l0p25_nodump_gpu89.py'

_gs3c_class_names = [
    'airport',
    'baseball-diamond',
    'basketball-court',
    'bridge',
    'container-crane',
    'ground-track-field',
    'harbor',
    'helicopter',
    'helipad',
    'large-vehicle',
    'plane',
    'roundabout',
    'ship',
    'small-vehicle',
    'soccer-ball-field',
    'storage-tank',
    'swimming-pool',
    'tennis-court',
]

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            enable=True,
            class_area_priors_csv=
            'work_dirs/semantic_ambiguity_study_20260617/p4_prediction_level_scale_prior/trainval_ms_full_exclude_ssval_source_area_priors.csv',
            class_names=_gs3c_class_names,
            domain_mode='ovd',
            mode='continuous_logit_energy',
            z0=4.0,
            beta=1.38629436112,
            adapter_hidden=64,
            adapter_nonpositive_delta=True,
            preserve_s3c_guard=True,
            guard_lambda=0.25)))

work_dir = 'work_dirs/semantic_ambiguity_study_20260617/p4_gs3c_g1_pre_nms_z4_beta1p386_nodump_gpu89'
