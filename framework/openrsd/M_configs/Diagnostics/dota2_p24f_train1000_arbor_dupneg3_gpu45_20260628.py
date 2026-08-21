_base_ = './dota2_p24a_train1000_arbor_replay_gpu45_20260628.py'

# P24F: strengthen duplicate hard-negative matching. P22B's train1000 AP is
# held back by very large detection counts for some classes, so this tests a
# stronger duplicate-objectness suppression schedule without changing the
# inference path.
work_dir = (
    'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/'
    'p24f_arbor_dupneg3_b16x2_seed3407')

model = dict(
    bbox_head=dict(
        matching_duplicate_neg_topk=3,
        matching_duplicate_neg_loss_weight=0.25,
        matching_duplicate_neg_warmup_iters=120,
        matching_duplicate_neg_decay_start_iters=440,
        matching_duplicate_neg_decay_end_iters=520,
        matching_duplicate_neg_decay_final_mult=0.25,
        matching_duplicate_neg_max_cost=6.0))
