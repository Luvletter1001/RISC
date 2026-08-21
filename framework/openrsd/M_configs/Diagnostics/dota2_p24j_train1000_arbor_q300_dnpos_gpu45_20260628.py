_base_ = './dota2_p24b_train1000_arbor_q300_gpu45_20260628.py'

work_dir = 'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/p24j_arbor_q300_dnpos_b16x2_seed3407'

model = dict(
    type='P15DNStyleQueryPosEncoderBypassSupportConditionedOrientedDINO',
    dn_style_query_pos_scale=0.5,
    bbox_head=dict(temperature=10000),
)

