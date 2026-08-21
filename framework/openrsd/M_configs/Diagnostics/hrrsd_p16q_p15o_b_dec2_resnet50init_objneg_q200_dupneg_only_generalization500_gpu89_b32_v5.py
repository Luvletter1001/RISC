_base_ = './hrrsd_p16j_p15o_b_dec2_resnet50init_objneg_q200_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16q_p15o_b_dec2_resnet50init_objneg_q200_dupneg_only_generalization500_gpu89_b32_v5')

# P16Q ablates P16O by keeping only matching-aware duplicate objectness
# negatives. It avoids the global count constraint and tests whether low-cost
# unmatched candidates are the main false-positive source.
model = dict(
    bbox_head=dict(
        duplicate_rank_topk=0,
        duplicate_rank_loss_weight=0.0,
        matching_duplicate_neg_topk=2,
        matching_duplicate_neg_loss_weight=0.20,
        matching_duplicate_neg_warmup_iters=120,
        matching_duplicate_neg_max_cost=6.0,
        cardinality_loss_weight=0.0))

randomness = dict(seed=3407, deterministic=False)
