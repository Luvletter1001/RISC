_base_ = './hrrsd_rtmdet_l_dota_init_internal_p14c_class_transport_mini.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'mini_p14j_one_to_many')

model = dict(
    bbox_head=dict(
        type='P14JOneToManyQueryTransportHead',
        aux_otm_loss_weight=0.75,
        aux_otm_topk_per_gt=3))
