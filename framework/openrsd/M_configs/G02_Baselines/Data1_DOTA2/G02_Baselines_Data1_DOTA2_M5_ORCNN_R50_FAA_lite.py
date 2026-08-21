_base_ = ['./G02_Baselines_Data1_DOTA2_M5_ORCNN_R50_FAA.py']

train_batch_size = 1
train_dataloader = dict(batch_size=train_batch_size)

default_hooks = dict(logger=dict(interval=1))

model = dict(
    neck=dict(
        fusion_modes=['faa', 'add', 'add'],
        fam_cfg=dict(m=7, c_mid=2)))
