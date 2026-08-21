_base_ = './hrrsd_rtmdet_l_dota_init_internal_p14b_refbox_transport_mini.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'mini_p14c_class_transport')

model = dict(
    bbox_head=dict(
        type='P14CClassConsistentQueryTransportHead',
        transport_rank_loss_weight=1.0,
        ranking_margin=0.08))
