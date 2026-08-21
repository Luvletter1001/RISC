_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1.py'
]

t6_parent = '8-T2-R-B1-E12'
t6_only_scientific_delta = 'fixed_size_rare_positive_exposure_4x'
rare_repeat_factor = 4
rare_unique_train_images = 36
rare_train_exposures = 144
replaced_empty_images = 108

rare_subset_root = (
    '/data1/zcy/OV-CapFlow/work_dirs/dotav2_cleanstart/'
    'subsets/seed20260715_rare4x/train/')

train_dataloader = dict(
    dataset=dict(
        data_root=rare_subset_root,
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/')),
    batch_sampler=dict(
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            's1_grouped_scale1024_seed20260716_gpu89_batch1_'
            'rare4x_sampler.json')))
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    's1_grouped_scale1024_seed20260716_gpu89_batch1_rare4x')
resume = False
