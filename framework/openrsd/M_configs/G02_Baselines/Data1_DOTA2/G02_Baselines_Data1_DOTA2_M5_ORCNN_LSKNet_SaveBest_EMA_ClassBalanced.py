_base_ = ['./G02_Baselines_Data1_DOTA2_M5_ORCNN_LSKNet_SaveBest_EMA.py']

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=0.02,
        dataset={{_base_.train_dataloader.dataset}}))

work_dir = 'work_dirs/lsknet_dotav2_ss_orcnn_bs2_savebest_ema_classbalanced'
