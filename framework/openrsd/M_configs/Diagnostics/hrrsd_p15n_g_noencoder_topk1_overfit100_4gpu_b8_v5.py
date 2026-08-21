_base_ = './hrrsd_p15l_c_topk1_class_balanced_query_overfit100_4gpu_b8_v5.py'

find_unused_parameters = True

work_dir = (
    'work_dirs/p15n_encoder_attitude_hrrsd_20260624/'
    'p15n_g_noencoder_topk1_overfit100_4gpu_b8_v5')

# P15N-G removes the encoder under the strongest existing P15L-C query
# initializer. This isolates whether no-encoder is only weak because P15N-B/D
# used the lower P15L-E base, or whether encoder context is intrinsically
# needed even when query selection is strong.
model = dict(
    type='P15EncoderBypassSupportConditionedOrientedDINO',
    encoder=dict(
        num_layers=1,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))),
    bypass_add_pos_to_memory=False)

randomness = dict(seed=3407, deterministic=False)
