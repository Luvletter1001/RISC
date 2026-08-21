_base_ = './hrrsd_p15o_b_dec2_noencoder_topk1_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15p_dnpos_decoder_light_hrrsd_20260624/'
    'p15p_a_dec2_dnpos_noencoder_topk1_overfit100_gpu4_b8_v5')

# P15P-A keeps P15O-B's decoder=2/no-encoder/topk1 strict-E2E setting, and
# adds a DN/DAB-style initial reference-box positional prior to query content
# before the rotated DINO decoder.
model = dict(
    type='P15DNStyleQueryPosEncoderBypassSupportConditionedOrientedDINO',
    dn_style_query_pos_scale=0.5,
    dn_style_query_pos_temperature=10000)

randomness = dict(seed=3407, deterministic=False)
