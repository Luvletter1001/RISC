_base_ = './hrrsd_p15l_class_balanced_query_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15l_class_balanced_query_hrrsd_20260624/'
    'overfit100_4gpu_b8_v5_b_seed3407')

# P15L-B is a repeat confirmation of P15L-A. Keep the method, dataset,
# strict-E2E inference path, and batch protocol unchanged; only change seed and
# output directory to test reproducibility.
randomness = dict(seed=3407, deterministic=False)
