_base_ = './hrrsd_p15_oriented_dino_overfit.py'

work_dir = 'work_dirs/p15b_scod_hrrsd_20260623/overfit20'

model = dict(
    bbox_head=dict(
        type='P15BOrientedDINOSetHead',
        bg_cls_weight=0.05,
        loss_obj_weight=1.0))
