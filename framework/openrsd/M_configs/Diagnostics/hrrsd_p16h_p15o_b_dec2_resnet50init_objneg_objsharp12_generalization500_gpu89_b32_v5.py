_base_ = './hrrsd_p16d_p15o_b_dec2_resnet50init_objneg_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16h_p15o_b_dec2_resnet50init_objneg_objsharp12_generalization500_gpu89_b32_v5'
)

# P16H keeps P16D's validated ImageNet init and objectness/background losses.
# P16G's 1.5 objectness power is a strong sharpening probe; this milder 1.2
# setting tests whether ranking calibration helps without starving early query
# diversity.
model = dict(
    bbox_head=dict(
        topk_objectness_power=1.2,
        matching_objectness_power=1.2))

randomness = dict(seed=3407, deterministic=False)
