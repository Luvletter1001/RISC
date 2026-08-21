_base_ = './dior_r_rtmdet_l_dota_init_gs3c_g1_light_eval_gpu0189.py'

work_dir = (
    'work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/'
    'eval_epoch3_plus3_g2_fitted_network_path_full')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            mode='logit_adapter',
            z0=4.0,
            # Isolate fitted G2 adapter; base G1 energy is disabled here.
            beta=0.0,
            adapter_hidden=64,
            adapter_nonpositive_delta=True,
            adapter_max_delta_abs=0.25,
            adapter_checkpoint=(
                'work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/'
                'g2_fitted_delta_target_z4_d0p25/adapter.pth'),
            adapter_checkpoint_strict=True)))
