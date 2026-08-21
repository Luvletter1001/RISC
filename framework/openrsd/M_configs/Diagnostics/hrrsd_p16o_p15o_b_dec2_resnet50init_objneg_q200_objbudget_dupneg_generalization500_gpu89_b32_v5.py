_base_ = './hrrsd_p16j_p15o_b_dec2_resnet50init_objneg_q200_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16o_p15o_b_dec2_resnet50init_objneg_q200_objbudget_dupneg_generalization500_gpu89_b32_v5')

# P16O moves duplicate control from a post-matching ranking hinge into
# objectness supervision itself. Keep P16J's strict no-NMS q200 recipe, then:
# 1) penalize low-cost unmatched duplicate candidates as hard objectness
#    negatives;
# 2) align summed query objectness with the number of GT instances.
model = dict(
    bbox_head=dict(
        duplicate_rank_topk=0,
        duplicate_rank_loss_weight=0.0,
        matching_duplicate_neg_topk=2,
        matching_duplicate_neg_loss_weight=0.20,
        matching_duplicate_neg_warmup_iters=120,
        matching_duplicate_neg_max_cost=6.0,
        cardinality_loss_weight=0.05,
        cardinality_warmup_iters=120))

randomness = dict(seed=3407, deterministic=False)
