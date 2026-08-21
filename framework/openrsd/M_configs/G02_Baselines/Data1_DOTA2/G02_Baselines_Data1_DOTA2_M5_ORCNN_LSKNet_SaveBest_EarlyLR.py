_base_ = ['./G02_Baselines_Data1_DOTA2_M5_ORCNN_LSKNet.py']

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=4,
        save_best='dota/mAP',
        rule='greater',
        max_keep_ckpts=3))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[4, 8],
        gamma=0.1)
]

work_dir = 'work_dirs/lsknet_dotav2_ss_orcnn_bs2_savebest_earlylr'
