_base_ = './hrrsd_p15l_e_one2many_primary_overfit100_4gpu_b8_v5.py'

find_unused_parameters = True

work_dir = (
    'work_dirs/p15n_encoder_attitude_hrrsd_20260624/'
    'p15n_d_noencoder_pos_e_base_overfit100_4gpu_b8_v5')

# P15N-D keeps the no-encoder bypass but adds the flattened positional/level
# encoding into memory. This tests whether P15N-B's drop comes from losing
# encoder context or simply from giving the proposal/query initializer raw neck
# features without position.
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
