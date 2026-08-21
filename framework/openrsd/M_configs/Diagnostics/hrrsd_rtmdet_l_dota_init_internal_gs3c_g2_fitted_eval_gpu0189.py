_base_ = './hrrsd_rtmdet_l_dota_init_internal_gs3c_g2_adapter_zero_eval_gpu0189.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'eval_epoch3_g2_fitted_network_path_full')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            mode='logit_adapter',
            z0=4.0,
            # Isolate fitted G2 adapter; base G1 energy is disabled here.
            beta=0.0,
            adapter_hidden=64,
            adapter_nonpositive_delta=True,
            adapter_max_delta_abs=0.5,
            adapter_min_abs_z=4.0,
            adapter_checkpoint=(
                'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
                'g2_fitted_scale_wrong_z4_d0p5/adapter.pth'),
            adapter_checkpoint_strict=True)))
