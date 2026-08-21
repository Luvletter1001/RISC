_base_ = './dota2_p24a_train1000_arbor_replay_gpu45_20260628.py'

work_dir = 'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/p24n_arbor_encoder2_b16x2_seed3407'

model = dict(
    encoder=dict(num_layers=2),
)

