_base_ = './dota2_p24a_train1000_arbor_replay_gpu45_20260628.py'

# P24B: raise the strict set size from 200 to 300 queries. The hypothesis is
# that DOTA2 train1000 has enough dense small objects that extra fixed queries
# can improve recall without returning to dense-head/NMS behavior.
work_dir = (
    'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/'
    'p24b_arbor_q300_b16x2_seed3407')

model = dict(
    num_queries=300,
    bbox_head=dict(num_queries=300, max_per_img=300),
    test_cfg=dict(max_per_img=300))
