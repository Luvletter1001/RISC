_base_ = './hrrsd_p15l_c_topk1_class_balanced_query_overfit100_4gpu_b8_v5.py'

find_unused_parameters = True

work_dir = (
    'work_dirs/p15n_encoder_attitude_hrrsd_20260624/'
    'p15n_h_noencoder_pos_topk1_overfit100_4gpu_b8_v5')

# P15N-H adds positional/level encoding to the no-encoder topk1 variant. If it
# recovers over P15N-G, the bypass loses mostly localization/level signal; if
# it stays below encoder1/2, the encoder is doing necessary context mixing.
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
    bypass_add_pos_to_memory=True)

randomness = dict(seed=3407, deterministic=False)
