_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_d13n_candidate.py'
]

d13n_role = 'raw-candidate'
d13n_mouth = 'raw13833'
d13n_test_only = True
d13n_master_port = 29844
d13n_checkpoint_arg_required = True
d13n_checkpoint_epoch = 12
d13n_inference_groups = 1
d13n_output_queries = 600
d13n_audit_path = (
    'work_dirs/dotav2_cleanstart/audits/'
    'd13n_raw13833_candidate_gpu56789_epoch_{epoch:02d}.json')

val_dataloader = dict(
    dataset=dict(
        data_root='/data1/zcy/datasets/DOTA2_1024_500/',
        ann_file='ss_val/annfiles/',
        data_prefix=dict(img_path='ss_val/images/'),
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=True))
test_dataloader = val_dataloader

train_cfg = None
train_dataloader = None
optim_wrapper = None
param_scheduler = None
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    'd13n_raw13833_candidate_gpu56789')
