_base_ = './hrrsd_p16b_p15o_b_dec2_resnet50init_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16e_p15o_b_dec2_resnet50init_objneg_mild_generalization500_gpu89_b32_v5')

# Mild version of P16D. P16D intentionally stresses unmatched-query
# objectness/background supervision; if that destabilizes early learning, this
# keeps the same hypothesis but uses a smaller step.
model = dict(
    bbox_head=dict(
        bg_cls_weight=0.15,
        loss_obj_weight=1.5))

randomness = dict(seed=3407, deterministic=False)
