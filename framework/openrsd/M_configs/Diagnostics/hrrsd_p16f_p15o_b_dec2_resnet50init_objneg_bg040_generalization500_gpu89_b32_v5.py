_base_ = './hrrsd_p16d_p15o_b_dec2_resnet50init_objneg_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16f_p15o_b_dec2_resnet50init_objneg_bg040_generalization500_gpu89_b32_v5')

# P16F keeps P16D's validated ImageNet initialization and objectness loss
# weight. Single new variable: increase unmatched-query/background weight from
# 0.25 to 0.40. The target is the remaining strict no-NMS failure mode: too
# many high-ranked foreground false positives. The inference contract is
# unchanged: no dense detection head, no NMS, no postprocess fallback.
model = dict(
    bbox_head=dict(
        bg_cls_weight=0.40))

randomness = dict(seed=3407, deterministic=False)
