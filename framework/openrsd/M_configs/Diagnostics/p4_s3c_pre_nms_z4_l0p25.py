_base_ = '../../work_dirs/dotav2_p4_lowtext_lser_sise_full9772_gpu67_20260617_1600/test_work/focus_ovd_a10_mess_fourier_dual_text_gpu67_lowtext_best_bs1.py'

_s3c_class_names = [
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
        scale_semantic_calibration=dict(
            enable=True,
            class_area_priors_csv=
            'work_dirs/semantic_ambiguity_study_20260617/p4_prediction_level_scale_prior/trainval_ms_full_exclude_ssval_source_area_priors.csv',
            class_names=_s3c_class_names,
            z_margin=4.0,
            downweight_lambda=0.25,
            mode='score_multiply',
            min_score=0.0,
            dump_topk_jsonl=
            'work_dirs/semantic_ambiguity_study_20260617/p4_s3c_pre_nms_z4_l0p25_gpu67/dense_topk_pre_nms.jsonl',
            dump_topk_k=20)))

launcher = 'pytorch'
work_dir = 'work_dirs/semantic_ambiguity_study_20260617/p4_s3c_pre_nms_z4_l0p25_gpu67'
