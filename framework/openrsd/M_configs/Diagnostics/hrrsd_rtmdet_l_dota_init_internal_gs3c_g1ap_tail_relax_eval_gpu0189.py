_base_ = './hrrsd_rtmdet_l_dota_init_internal_gs3c_g1_z4_b1p386_eval_gpu0189.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'eval_epoch3_g1ap_tail_relax_full')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            z0=4.0,
            beta=1.38629436112,
            classwise_z0={
                'airplane': 5.0,
                'ship': 5.0,
                'harbor': 5.0,
                'parking': 5.0,
                'ground': 5.0,
                'storage': 5.0,
                'bridge': 5.0,
                'baseball': 5.0,
            },
            classwise_beta={
                'airplane': 0.25,
                'ship': 0.25,
                'harbor': 0.25,
                'parking': 0.25,
                'ground': 0.25,
                'storage': 0.25,
                'bridge': 0.25,
                'baseball': 0.25,
            })))
