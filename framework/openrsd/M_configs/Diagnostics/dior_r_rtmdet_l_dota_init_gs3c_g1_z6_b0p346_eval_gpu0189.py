_base_ = './dior_r_rtmdet_l_dota_init_gs3c_g1_light_eval_gpu0189.py'

work_dir = (
    'work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/'
    'eval_g1_z6_b0p346_full')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            z0=6.0,
            beta=0.34657359028)))
