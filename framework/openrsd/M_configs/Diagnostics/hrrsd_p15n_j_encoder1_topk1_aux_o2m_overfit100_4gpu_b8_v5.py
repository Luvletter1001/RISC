_base_ = './hrrsd_p15l_c_topk1_class_balanced_query_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15n_encoder_attitude_hrrsd_20260624/'
    'p15n_j_encoder1_topk1_aux_o2m_overfit100_4gpu_b8_v5')

# P15N-J keeps the light one-layer encoder and topk1 initializer, then restores
# the low-weight one-to-many primary stabilizer. This tests whether the aux O2M
# signal only hurt the topk2 base or can help the lighter encoder converge.
model = dict(
    encoder=dict(
        num_layers=1,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))),
    bbox_head=dict(
        aux_one2many_topk=1,
        aux_one2many_loss_weight=0.10,
        aux_one2many_warmup_iters=80))

randomness = dict(seed=3407, deterministic=False)
