_base_ = './hrrsd_p16d_p15o_b_dec2_resnet50init_objneg_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16k_p15o_b_dec2_resnet50init_objneg_duprank_generalization500_gpu89_b32_v5')

# P16K keeps P16D's strict E2E q300 inference path unchanged:
# fixed query set, no dense head, no NMS, no score-threshold fallback.
# The only new signal is training-time duplicate ranking. The matched
# one-to-one query for each GT is encouraged to outrank nearby unmatched
# duplicate candidates, targeting the high det-count false-positive failure.
model = dict(
    bbox_head=dict(
        duplicate_rank_topk=2,
        duplicate_rank_loss_weight=0.15,
        duplicate_rank_margin=0.15,
        duplicate_rank_warmup_iters=120,
        duplicate_rank_max_cost=2.0))

randomness = dict(seed=3407, deterministic=False)
