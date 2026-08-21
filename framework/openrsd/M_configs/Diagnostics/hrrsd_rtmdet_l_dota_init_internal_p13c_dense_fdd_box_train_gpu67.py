_base_ = './hrrsd_rtmdet_l_dota_init_internal_p13c_dense_fdd_safe_train_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus3_p13c_dense_fdd_box')

model = dict(
    bbox_head=dict(
        p13_decoder=dict(
            bbox_delta_weight=0.05,
            angle_delta_weight=0.05)))
