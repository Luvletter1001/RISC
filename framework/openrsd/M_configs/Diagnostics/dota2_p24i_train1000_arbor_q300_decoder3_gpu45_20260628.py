_base_ = './dota2_p24b_train1000_arbor_q300_gpu45_20260628.py'

work_dir = 'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/p24i_arbor_q300_decoder3_b16x2_seed3407'

model = dict(
    decoder=dict(num_layers=3),
)

