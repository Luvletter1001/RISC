_base_ = './hrrsd_rtmdet_l_dota_init_internal_p13e_k100_seed_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus3_p13e_k200_seed_strong')

model = dict(
    bbox_head=dict(
        num_queries=200,
        num_decoder_layers=3,
        max_per_img=200,
        support_scale=1.5,
        logit_scale_init=3.0,
        cls_loss_weight=3.0,
        quality_loss_weight=1.5,
        seed_loss_weight=0.5,
        match_cls_cost=3.0,
        score_thr=0.03))
