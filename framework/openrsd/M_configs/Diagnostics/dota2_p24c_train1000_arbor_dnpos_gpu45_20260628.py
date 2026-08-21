_base_ = './dota2_p24a_train1000_arbor_replay_gpu45_20260628.py'

# P24C: add a DAB/DN-style initial query position prior. This keeps the
# no-encoder strict-E2E path but gives matching and denoising queries explicit
# geometry before decoder refinement.
work_dir = (
    'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/'
    'p24c_arbor_dnpos_b16x2_seed3407')

model = dict(
    type='P15DNStyleQueryPosEncoderBypassSupportConditionedOrientedDINO',
    dn_style_query_pos_scale=0.5,
    dn_style_query_pos_temperature=10000)
