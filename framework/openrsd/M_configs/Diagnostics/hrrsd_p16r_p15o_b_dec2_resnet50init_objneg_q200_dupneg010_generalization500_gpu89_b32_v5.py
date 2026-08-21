_base_ = './hrrsd_p16j_p15o_b_dec2_resnet50init_objneg_q200_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16r_p15o_b_dec2_resnet50init_objneg_q200_dupneg010_generalization500_gpu89_b32_v5')

# P16R is the conservative backup for P16Q: keep the same matching-aware
# duplicate hard-negative target, but halve its weight to test late over-suppression.
model = dict(
    bbox_head=dict(
        duplicate_rank_topk=0,
        duplicate_rank_loss_weight=0.0,
        matching_duplicate_neg_topk=2,
        matching_duplicate_neg_loss_weight=0.10,
        matching_duplicate_neg_warmup_iters=120,
        matching_duplicate_neg_max_cost=6.0,
        cardinality_loss_weight=0.0))

randomness = dict(seed=3407, deterministic=False)
