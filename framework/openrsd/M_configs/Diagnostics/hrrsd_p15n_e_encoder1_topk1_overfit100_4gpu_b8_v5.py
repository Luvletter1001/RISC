_base_ = './hrrsd_p15l_c_topk1_class_balanced_query_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15n_encoder_attitude_hrrsd_20260624/'
    'p15n_e_encoder1_topk1_overfit100_4gpu_b8_v5')

# P15N-E combines the single-layer encoder with the stronger P15L-C query
# initializer family. P15L-C was the best overfit100 signal so far, so this
# checks whether a light encoder can keep that quality while avoiding the full
# 6-layer cost.
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
