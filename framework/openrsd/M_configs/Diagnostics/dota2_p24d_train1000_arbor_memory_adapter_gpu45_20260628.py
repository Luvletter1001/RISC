_base_ = './dota2_p24a_train1000_arbor_replay_gpu45_20260628.py'

find_unused_parameters = True

# P24D: replace pure encoder bypass memory with a cheap token-wise adapter
# after adding position/level features. It tests whether a small model-side
# projection can recover context lost by the bypassed deformable encoder.
work_dir = (
    'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/'
    'p24d_arbor_memory_adapter_pos_b16x2_seed3407')

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
