_base_ = './hrrsd_p15l_c_topk1_class_balanced_query_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15n_encoder_attitude_hrrsd_20260624/'
    'p15n_f_encoder2_topk1_overfit100_4gpu_b8_v5')

# P15N-F tests whether P15N-A's 2-layer encoder becomes stronger when paired
# with the P15L-C topk1 class-balanced query initializer instead of P15L-E's
# one-to-many primary stabilizer.
model = dict(
    encoder=dict(
        num_layers=2,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))))

randomness = dict(seed=3407, deterministic=False)
