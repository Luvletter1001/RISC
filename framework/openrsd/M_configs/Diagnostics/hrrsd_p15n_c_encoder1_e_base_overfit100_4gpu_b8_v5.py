_base_ = './hrrsd_p15l_e_one2many_primary_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15n_encoder_attitude_hrrsd_20260624/'
    'p15n_c_encoder1_e_base_overfit100_4gpu_b8_v5')

# P15N-C is the direct midpoint between P15N-A and P15N-B on the same P15L-E
# base. It keeps strict-E2E inference, batch, data, eval interval, and
# one-to-many stabilizer unchanged while reducing the deformable encoder to a
# single layer.
model = dict(
    encoder=dict(
        num_layers=1,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))))

randomness = dict(seed=3407, deterministic=False)
