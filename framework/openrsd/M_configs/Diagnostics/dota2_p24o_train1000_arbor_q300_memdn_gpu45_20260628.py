_base_ = './dota2_p24b_train1000_arbor_q300_gpu45_20260628.py'

find_unused_parameters = True

work_dir = (
    'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/'
    'p24o_arbor_q300_memory_adapter_dnpos_b16x2_seed3407')

# P24O combines two structural changes in the strict-E2E Arbor path:
# token-wise positional memory adaptation and DN/DAB-style initial query
# position priors. It tests whether memory recalibration and geometry-aware
# query content are complementary under the q300 regime.
model = dict(
    type='P24MemoryAdapterDNStyleQueryPosSupportConditionedOrientedDINO',
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
    memory_adapter_norm=True,
    dn_style_query_pos_scale=0.5,
    dn_style_query_pos_temperature=10000)

