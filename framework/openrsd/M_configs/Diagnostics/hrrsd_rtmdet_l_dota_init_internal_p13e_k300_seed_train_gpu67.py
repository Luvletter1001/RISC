_base_ = './hrrsd_rtmdet_l_dota_init_internal_p13e_k100_seed_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus3_p13e_k300_seed')

model = dict(
    bbox_head=dict(
        num_queries=300,
        max_per_img=300,
        score_thr=0.02))
