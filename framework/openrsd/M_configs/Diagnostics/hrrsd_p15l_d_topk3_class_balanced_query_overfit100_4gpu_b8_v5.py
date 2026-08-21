_base_ = './hrrsd_p15l_b_repeat_class_balanced_query_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15l_class_balanced_query_hrrsd_20260624/'
    'overfit100_4gpu_b8_v5_d_topk3_seed3407')

# P15L-D increases the train-time per-class quota to test whether extra weak
# class query exposure improves ship/vehicle without changing inference.
model = dict(
    bbox_head=dict(
        train_class_balanced_topk_per_class=3))
