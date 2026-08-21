_base_ = './hrrsd_p16j_p15o_b_dec2_resnet50init_objneg_q200_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16m_p15o_b_dec2_resnet50init_objneg_q200_duprank_cost6_generalization500_gpu89_b32_v5')

# P16M fixes the failed P16L trigger condition.
# P16L kept loss_duplicate_rank at 0.0000 through e12 because max_cost=2.0
# selected no duplicate candidates. Keep the same strict-E2E q200 recipe and
# only widen the training-time duplicate candidate window.
model = dict(
    bbox_head=dict(
        duplicate_rank_topk=2,
        duplicate_rank_loss_weight=0.15,
        duplicate_rank_margin=0.15,
        duplicate_rank_warmup_iters=120,
        duplicate_rank_max_cost=6.0))

randomness = dict(seed=3407, deterministic=False)
