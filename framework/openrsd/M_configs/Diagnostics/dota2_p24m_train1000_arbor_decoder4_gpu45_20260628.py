_base_ = './dota2_p24a_train1000_arbor_replay_gpu45_20260628.py'

work_dir = 'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/p24m_arbor_decoder4_b16x2_seed3407'

model = dict(
    decoder=dict(num_layers=4),
)

