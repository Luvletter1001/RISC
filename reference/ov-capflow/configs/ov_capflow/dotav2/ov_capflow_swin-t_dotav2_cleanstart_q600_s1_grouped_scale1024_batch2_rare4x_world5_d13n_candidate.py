_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world5_d13n_control.py'
]

d13n_role = 'candidate'
d13n_master_port = 29841
d13n_audit_path = (
    'work_dirs/dotav2_cleanstart/audits/'
    'd13n_world5_candidate_seed20260716_gpu56789_epoch_{epoch:02d}.json')

model = dict(bbox_head=dict(existence_loss_weight=1.0))
train_dataloader = dict(
    batch_sampler=dict(audit_path=d13n_audit_path))
custom_hooks = [dict(type='D13NParentEvalModeHook', role='candidate')]
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    'd13n_world5_candidate_seed20260716_gpu56789_batch2')
