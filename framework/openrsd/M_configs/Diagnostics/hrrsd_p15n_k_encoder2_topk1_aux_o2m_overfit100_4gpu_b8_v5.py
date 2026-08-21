_base_ = './hrrsd_p15l_c_topk1_class_balanced_query_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15n_encoder_attitude_hrrsd_20260624/'
    'p15n_k_encoder2_topk1_aux_o2m_overfit100_4gpu_b8_v5')

# P15N-K pairs the current best light encoder depth candidate with topk1 and
# low-weight aux O2M. It is a direct refinement of P15N-F if F is promising but
# still below the 6-layer P15L-C anchor.
model = dict(
    encoder=dict(
        num_layers=2,
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
