_base_ = '../../work_dirs/dotav2_p4_lowtext_lser_sise_full9772_gpu67_20260617_1600/test_work/focus_ovd_a10_mess_fourier_dual_text_gpu67_lowtext_best_bs1.py'

model = dict(
    bbox_head=dict(
        scale_semantic_calibration=dict(enable=False)))

work_dir = 'work_dirs/semantic_ambiguity_study_20260617/p4_pre_nms_no_s3c_gpu01'
