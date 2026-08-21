_base_ = './hrrsd_p16d_p15o_b_dec2_resnet50init_objneg_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16j_p15o_b_dec2_resnet50init_objneg_q200_generalization500_gpu89_b32_v5')

# P16J is the follow-up to P16I. Query=150 improved early precision but
# dropped by epoch20, indicating recall/class coverage was too narrow. Keep
# P16D's successful initialization and loss calibration, but use 200 queries
# as a middle point between over-dense 300 and recall-limited 150.
model = dict(
    num_queries=200,
    bbox_head=dict(
        num_queries=200,
        max_per_img=200),
    test_cfg=dict(max_per_img=200))

randomness = dict(seed=3407, deterministic=False)
