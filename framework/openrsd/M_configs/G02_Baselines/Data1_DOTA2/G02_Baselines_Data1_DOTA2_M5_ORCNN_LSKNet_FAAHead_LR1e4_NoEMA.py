_base_ = ['./G02_Baselines_Data1_DOTA2_M5_ORCNN_LSKNet_FAAHead.py']

custom_hooks = [dict(type='mmdet.NumClassCheckHook')]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2))

work_dir = 'work_dirs/faahead_dotav2_ss_lsknet_bs2_lr1e4_noema'
