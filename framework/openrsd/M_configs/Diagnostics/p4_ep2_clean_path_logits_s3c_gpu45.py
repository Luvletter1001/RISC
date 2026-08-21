_base_ = './p4_s3c_pre_nms_z4_l0p25_gpu45.py'

model = dict(
    bbox_head=dict(
        scale_semantic_calibration=dict(
            ep2_path_probe=dict(
                enable=True,
                output_csv=(
                    'work_dirs/semantic_ambiguity_study_20260617/'
                    'p4_ep2_clean_path_logits_s3c_gpu45/'
                    'ep2_pre_nms_raw_logits.csv'),
                # Format is gt_class->hardneg_class. The historical
                # "small-vehicle->tennis-court" error is therefore probed as
                # tennis-court GT against small-vehicle hard-negative logits.
                class_pairs=[
                    ('tennis-court', 'small-vehicle'),
                    ('plane', 'small-vehicle'),
                    ('harbor', 'ship'),
                    ('ship', 'harbor'),
                ],
                max_locations_per_level=128))))

work_dir = (
    'work_dirs/semantic_ambiguity_study_20260617/'
    'p4_ep2_clean_path_logits_s3c_gpu45')
