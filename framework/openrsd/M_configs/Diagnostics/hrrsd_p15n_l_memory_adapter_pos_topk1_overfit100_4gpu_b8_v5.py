_base_ = './hrrsd_p15l_c_topk1_class_balanced_query_overfit100_4gpu_b8_v5.py'

find_unused_parameters = True

work_dir = (
    'work_dirs/p15n_encoder_attitude_hrrsd_20260624/'
    'p15n_l_memory_adapter_pos_topk1_overfit100_4gpu_b8_v5')

# P15N-L bypasses the deformable encoder but applies a cheap token-wise memory
# adapter after adding positional/level memory. It tests whether the encoder's
# benefit can be approximated by lightweight normalization/projection rather
# than cross-token attention.
model = dict(
    type='P15MemoryAdapterSupportConditionedOrientedDINO',
    encoder=dict(
        num_layers=1,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))),
    memory_adapter_add_pos=True,
    memory_adapter_hidden_channels=512,
    memory_adapter_residual_scale=0.5,
    memory_adapter_norm=True)

randomness = dict(seed=3407, deterministic=False)
