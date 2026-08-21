_base_ = './hrrsd_p15l_c_topk1_class_balanced_query_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15l_class_balanced_query_hrrsd_20260624/'
    'overfit100_4gpu_b8_v5_c2_topk1_seed91011')

# P15L-C2 repeats the promoted topk1 quota with a different seed. Keep method,
# data, batch, schedule, and strict-E2E inference unchanged.
randomness = dict(seed=91011, deterministic=False)
