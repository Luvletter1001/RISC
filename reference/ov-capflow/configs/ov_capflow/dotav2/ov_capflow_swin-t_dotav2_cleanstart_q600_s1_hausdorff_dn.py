_base_ = ['./ov_capflow_swin-t_dotav2_cleanstart_q600_s1_control.py']

model = dict(
    dn_cfg=dict(angle_noise_scale=0.25),
    bbox_head=dict(adaptive_dn_cfg=dict(enabled=True)),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.HungarianAssigner',
            match_costs=[
                dict(type='mmdet.BinaryFocalLossCost', weight=2.0),
                dict(
                    type='RBoxL1Cost', weight=5.0,
                    box_format='xywha'),
                dict(
                    type='GDCost', weight=2.0, loss_type='kld',
                    fun='log1p', tau=1, sqrt=False),
                dict(
                    type='HausdorffCost', weight=2.0,
                    box_format='xywha'),
            ]),
        dn_assigner=dict(
            type='DNGroupHungarianAssigner',
            match_costs=[
                dict(type='mmdet.BinaryFocalLossCost', weight=2.0),
                dict(
                    type='RBoxL1Cost', weight=5.0,
                    box_format='xywha'),
                dict(
                    type='HausdorffCost', weight=2.0,
                    box_format='xywha'),
            ])))
work_dir = 'work_dirs/dotav2_cleanstart/s1_hausdorff_dn'
