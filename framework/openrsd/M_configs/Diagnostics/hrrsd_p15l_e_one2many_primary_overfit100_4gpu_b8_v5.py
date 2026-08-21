_base_ = './hrrsd_p15l_b_repeat_class_balanced_query_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15l_e_one2many_primary_hrrsd_20260624/'
    'overfit100_4gpu_b8_v5_a')

# P15L-E keeps the P15L-B strict-E2E inference path unchanged. It only adds
# low-weight training-time mixed supervision on primary matching queries.
model = dict(
    bbox_head=dict(
        aux_one2many_topk=1,
        aux_one2many_loss_weight=0.10,
        aux_one2many_warmup_iters=80))

randomness = dict(seed=3407, deterministic=False)
