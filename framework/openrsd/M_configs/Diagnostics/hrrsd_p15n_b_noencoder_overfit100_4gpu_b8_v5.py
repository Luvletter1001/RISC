_base_ = './hrrsd_p15l_e_one2many_primary_overfit100_4gpu_b8_v5.py'

find_unused_parameters = True

work_dir = (
    'work_dirs/p15n_encoder_ablation_hrrsd_20260624/'
    'p15n_b_noencoder_overfit100_4gpu_b8_v5')

# P15N-B verifies the user's hypothesis directly: CLIP/DINO support semantics
# and the image backbone/neck may be sufficient, so the deformable encoder is
# bypassed. The encoder module is still constructible for config compatibility,
# but forward_encoder returns flattened neck memory directly.
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
