_base_ = ['./G02_Baselines_Data1_DOTA2_M5_ORCNN_LSKNet.py']

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=4,
        save_best='dota/mAP',
        rule='greater',
        max_keep_ckpts=3))

work_dir = 'work_dirs/lsknet_dotav2_ss_orcnn_bs2_savebest'
