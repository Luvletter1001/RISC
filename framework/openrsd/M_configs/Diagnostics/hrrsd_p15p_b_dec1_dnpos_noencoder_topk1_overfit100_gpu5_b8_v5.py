_base_ = './hrrsd_p15o_c_dec1_noencoder_topk1_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15p_dnpos_decoder_light_hrrsd_20260624/'
    'p15p_b_dec1_dnpos_noencoder_topk1_overfit100_gpu5_b8_v5')

# P15P-B is the one-layer decoder speed lower bound with the same DN/DAB-style
# initial reference positional prior as P15P-A.
model = dict(
    type='P15DNStyleQueryPosEncoderBypassSupportConditionedOrientedDINO',
    dn_style_query_pos_scale=0.5,
    dn_style_query_pos_temperature=10000)

randomness = dict(seed=3407, deterministic=False)
