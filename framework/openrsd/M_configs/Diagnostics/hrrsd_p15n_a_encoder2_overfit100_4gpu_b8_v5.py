_base_ = './hrrsd_p15l_e_one2many_primary_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15n_encoder_ablation_hrrsd_20260624/'
    'p15n_a_encoder2_overfit100_4gpu_b8_v5')

# P15N-A tests whether the 6-layer deformable encoder is overbuilt for the
# current OpenRSD/P15L pipeline. Keep P15L-E's strict-E2E output contract,
# support-conditioned query initializer, O2M training stabilizer, dataset,
# batch, and eval protocol unchanged.
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
