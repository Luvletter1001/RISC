_base_ = './hrrsd_p16j_p15o_b_dec2_resnet50init_objneg_q200_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16t_p15o_b_dec2_resnet50init_objneg_q200_dupneg020_latedecay_generalization500_gpu89_b32_v5')

# P16T keeps P16Q's strong early duplicate hard-negative signal, then decays it
# after the observed P16Q peak to test whether the later e60/e70 drop is caused
# by over-suppressing hard duplicate queries.
model = dict(
    bbox_head=dict(
        duplicate_rank_topk=0,
        duplicate_rank_loss_weight=0.0,
        matching_duplicate_neg_topk=2,
        matching_duplicate_neg_loss_weight=0.20,
        matching_duplicate_neg_warmup_iters=120,
        matching_duplicate_neg_decay_start_iters=440,
        matching_duplicate_neg_decay_end_iters=520,
        matching_duplicate_neg_decay_final_mult=0.5,
        matching_duplicate_neg_max_cost=6.0,
        cardinality_loss_weight=0.0))

randomness = dict(seed=3407, deterministic=False)
