_base_ = './hrrsd_p16j_p15o_b_dec2_resnet50init_objneg_q200_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16p_p15o_b_dec2_resnet50init_objneg_q200_cardinality_only_generalization500_gpu89_b32_v5')

# P16P ablates P16O by keeping only the objectness cardinality budget.
# Use this if P16O hurts early mAP and we need to separate global objectness
# calibration from duplicate hard-negative mining.
model = dict(
    bbox_head=dict(
        duplicate_rank_topk=0,
        duplicate_rank_loss_weight=0.0,
        matching_duplicate_neg_topk=0,
        matching_duplicate_neg_loss_weight=0.0,
        cardinality_loss_weight=0.05,
        cardinality_warmup_iters=120))

randomness = dict(seed=3407, deterministic=False)
