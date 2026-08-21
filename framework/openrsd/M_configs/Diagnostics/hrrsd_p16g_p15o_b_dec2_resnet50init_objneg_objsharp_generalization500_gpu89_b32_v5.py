_base_ = './hrrsd_p16d_p15o_b_dec2_resnet50init_objneg_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16g_p15o_b_dec2_resnet50init_objneg_objsharp_generalization500_gpu89_b32_v5')

# P16G keeps P16D's validated init and background/objectness supervision.
#
# Single new variable: sharpen the objectness prior used before the decoder
# and during Hungarian matching. This tests whether the remaining no-NMS
# failure is mainly posterior ranking noise. Inference still emits the same
# strict E2E set output: no dense detection head, no NMS, no postprocess fallback.
model = dict(
    bbox_head=dict(
        topk_objectness_power=1.5,
        matching_objectness_power=1.5))

randomness = dict(seed=3407, deterministic=False)
