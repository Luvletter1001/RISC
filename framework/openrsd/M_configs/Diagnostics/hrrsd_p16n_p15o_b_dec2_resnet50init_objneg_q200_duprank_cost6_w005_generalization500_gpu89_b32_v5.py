_base_ = './hrrsd_p16j_p15o_b_dec2_resnet50init_objneg_q200_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16n_p15o_b_dec2_resnet50init_objneg_q200_duprank_cost6_w005_generalization500_gpu89_b32_v5')

# P16N follows P16M's positive trigger fix but reduces early ranking pressure.
# P16M proves cost6 activates duplicate-rank, but e10 remains below P16J.
# Keep candidate coverage and lower only the training-time rank loss weight.
model = dict(
    bbox_head=dict(
        duplicate_rank_topk=2,
        duplicate_rank_loss_weight=0.05,
        duplicate_rank_margin=0.15,
        duplicate_rank_warmup_iters=120,
        duplicate_rank_max_cost=6.0))

randomness = dict(seed=3407, deterministic=False)
