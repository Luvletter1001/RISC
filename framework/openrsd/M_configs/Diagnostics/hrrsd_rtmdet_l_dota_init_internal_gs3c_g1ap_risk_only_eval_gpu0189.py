_base_ = './hrrsd_rtmdet_l_dota_init_internal_gs3c_g1_z4_b1p386_eval_gpu0189.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'eval_epoch3_g1ap_risk_only_full')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            z0=4.5,
            beta=0.25,
            classwise_z0={
                'tennis': 4.0,
                'vehicle': 4.0,
            },
            classwise_beta={
                'tennis': 1.38629436112,
                'vehicle': 1.38629436112,
            })))
