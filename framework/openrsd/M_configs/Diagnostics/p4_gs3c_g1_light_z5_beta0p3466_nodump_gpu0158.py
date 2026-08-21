_base_ = './p4_gs3c_g1_pre_nms_z4_beta1p386_nodump_gpu89.py'

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            z0=5.0,
            beta=0.34657359028,
            preserve_s3c_guard=True,
            guard_lambda=0.25)))

work_dir = 'work_dirs/semantic_ambiguity_study_20260617/p4_gs3c_g1_light_z5_beta0p3466_nodump_gpu0158'
