_base_ = ['./ov_capflow_swin-t_dotav2_cleanstart_q600_s1_control.py']

model = dict(
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='mmdet.BinaryFocalLossCost', weight=2.0),
                dict(
                    type='ChamferCost', weight=5.0,
                    box_format='xywha'),
                dict(
                    type='GDCost', weight=2.0, loss_type='kld',
                    fun='log1p', tau=1, sqrt=False),
            ])))

work_dir = 'work_dirs/dotav2_cleanstart/s1_chamfer'
