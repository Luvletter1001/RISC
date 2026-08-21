_base_ = './dior_r_rtmdet_l_dota_init_gs3c_g1_light_eval_gpu0189.py'

work_dir = (
    'work_dirs/gs3c_dior_r_rtmdetl_dota_init_20260619/'
    'eval_epoch3_plus3_g2_adapter_zero_full')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            mode='logit_adapter',
            z0=4.0,
            # Disable base G1 energy so this validates the zero-init G2
            # adapter path without changing baseline logits.
            beta=0.0,
            adapter_hidden=64,
            adapter_nonpositive_delta=True)))
