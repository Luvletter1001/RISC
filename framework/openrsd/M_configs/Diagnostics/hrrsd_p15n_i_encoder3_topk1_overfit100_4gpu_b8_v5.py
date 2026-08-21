_base_ = './hrrsd_p15l_c_topk1_class_balanced_query_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15n_encoder_attitude_hrrsd_20260624/'
    'p15n_i_encoder3_topk1_overfit100_4gpu_b8_v5')

# P15N-I is a fallback if P15N-F shows that two encoder layers are not enough
# under the strong topk1 query initializer. It checks whether the quality jump
# needs a third deformable encoder layer or the full six-layer stack.
model = dict(
    encoder=dict(
        num_layers=3,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))))

randomness = dict(seed=3407, deterministic=False)
