_base_ = './hrrsd_p16d_p15o_b_dec2_resnet50init_objneg_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16i_p15o_b_dec2_resnet50init_objneg_q150_generalization500_gpu89_b32_v5')

# P16I targets the remaining strict-E2E failure mode observed in P16D/F:
# fixed 300-query no-NMS output leaves too many duplicate/low-precision
# foreground candidates. Keep P16D's successful initialization and loss
# calibration, but reduce the architectural set size to 150 queries.
model = dict(
    num_queries=150,
    bbox_head=dict(
        num_queries=150,
        max_per_img=150),
    test_cfg=dict(max_per_img=150))

randomness = dict(seed=3407, deterministic=False)
