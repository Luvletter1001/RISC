import os

_base_ = './p4_s3c_pre_nms_z4_l0p25_gpu45.py'

_ep2_output_csv = os.environ.get(
    'EP2_PATH_PROBE_CSV',
    'work_dirs/semantic_ambiguity_study_20260617/'
    'ep2_clean_path_probe_p4/ep2_pre_nms_raw_logits.csv')
_ep2_max_locations = int(os.environ.get(
    'EP2_PATH_PROBE_MAX_LOCATIONS', '128'))

model = dict(
    bbox_head=dict(
        scale_semantic_calibration=dict(
            ep2_path_probe=dict(
                enable=True,
                output_csv=_ep2_output_csv,
                class_pairs=[
                    ('tennis-court', 'small-vehicle'),
                    ('plane', 'small-vehicle'),
                    ('harbor', 'ship'),
                    ('ship', 'harbor'),
                ],
                max_locations_per_level=_ep2_max_locations))))

work_dir = os.environ.get(
    'EP2_PATH_PROBE_WORK_DIR',
    'work_dirs/semantic_ambiguity_study_20260617/ep2_clean_path_probe_p4')
