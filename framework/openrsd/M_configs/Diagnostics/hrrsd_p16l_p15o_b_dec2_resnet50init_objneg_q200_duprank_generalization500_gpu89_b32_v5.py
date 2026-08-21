_base_ = './hrrsd_p16j_p15o_b_dec2_resnet50init_objneg_q200_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16l_p15o_b_dec2_resnet50init_objneg_q200_duprank_generalization500_gpu89_b32_v5')

# P16L isolates the next hypothesis after P16J:
# q200 is better than q300/q150 for the strict E2E path, but the remaining
# failure is duplicate/ranking noise under no-NMS inference. Keep P16J's
# query count and schedule; add only training-time duplicate ranking.
model = dict(
    bbox_head=dict(
        duplicate_rank_topk=2,
        duplicate_rank_loss_weight=0.15,
        duplicate_rank_margin=0.15,
        duplicate_rank_warmup_iters=120,
        duplicate_rank_max_cost=2.0))

randomness = dict(seed=3407, deterministic=False)
