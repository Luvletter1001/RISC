_base_ = './hrrsd_rtmdet_l_dota_init_internal_p14c_class_transport_mini.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'mini_p14k_groupwise_transport')

model = dict(
    bbox_head=dict(
        type='P14KGroupWiseQueryTransportHead',
        group_transport_loss_weight=0.5,
        query_group_count=2,
        inference_single_group=True,
        max_per_img=130))
