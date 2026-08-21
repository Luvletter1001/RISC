_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1.py'
]

t4_parent = '8-T2-R-B1-E12'
t4_only_scientific_delta = 'language_model_lr_mult=0.1'

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(language_model=dict(lr_mult=0.1))))
train_dataloader = dict(
    batch_sampler=dict(
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            's1_grouped_scale1024_seed20260716_gpu89_batch1_'
            'textlr1e5_sampler.json')))
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    's1_grouped_scale1024_seed20260716_gpu89_batch1_textlr1e5')
resume = False
