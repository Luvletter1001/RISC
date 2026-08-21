_base_ = './hrrsd_p16b_p15o_b_dec2_resnet50init_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16d_p15o_b_dec2_resnet50init_objneg_generalization500_gpu89_b32_v5')

# P16D tests the second root-cause hypothesis after P16B:
# ImageNet init fixes the collapse, but the strict no-NMS set output still
# emits many low-precision duplicate foreground predictions. Keep the E2E
# architecture unchanged and only strengthen unmatched-query/background
# supervision plus objectness calibration.
model = dict(
    bbox_head=dict(
        bg_cls_weight=0.25,
        loss_obj_weight=2.0))

randomness = dict(seed=3407, deterministic=False)
