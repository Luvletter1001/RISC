_base_ = './hrrsd_p15l_b_repeat_class_balanced_query_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15l_class_balanced_query_hrrsd_20260624/'
    'overfit100_4gpu_b8_v5_c_topk1_seed3407')

# P15L-C ablates the class-balanced query quota while keeping P15L-B's seed,
# data protocol, batch protocol, and strict-E2E inference path unchanged.
model = dict(
    bbox_head=dict(
        train_class_balanced_topk_per_class=1))
