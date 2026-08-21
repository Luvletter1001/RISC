_base_ = './p4_gs3c_g1_pre_nms_z4_beta1p386_nodump_gpu89.py'

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            mode='logit_adapter',
            adapter_hidden=64,
            adapter_nonpositive_delta=True,
            preserve_s3c_guard=True)))

work_dir = 'work_dirs/semantic_ambiguity_study_20260617/p4_gs3c_g2_adapter_zero_pre_nms_nodump_gpu89'
