_base_ = './dota2_p24b_train1000_arbor_q300_gpu45_20260628.py'

work_dir = 'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/p24k_arbor_q300_dupneg3_b16x2_seed3407'

model = dict(
    bbox_head=dict(
        matching_duplicate_neg_topk=3,
        matching_duplicate_neg_loss_weight=0.25,
        matching_duplicate_neg_warmup_iters=120,
        matching_duplicate_neg_decay_start=440,
        matching_duplicate_neg_decay_end=520,
        matching_duplicate_neg_final_mult=0.25,
        matching_duplicate_neg_max_cost=6.0,
    ),
)

