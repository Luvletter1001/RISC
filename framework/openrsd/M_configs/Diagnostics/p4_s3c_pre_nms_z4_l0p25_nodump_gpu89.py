_base_ = './p4_s3c_pre_nms_z4_l0p25.py'

model = dict(
    bbox_head=dict(
        scale_semantic_calibration=dict(
            dump_topk_jsonl=None)))

work_dir = 'work_dirs/semantic_ambiguity_study_20260617/p4_s3c_pre_nms_z4_l0p25_nodump_gpu89'
