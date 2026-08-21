_base_ = './dota2_p24a_train1000_arbor_replay_gpu45_20260628.py'

work_dir = 'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/p24g_arbor_lr1e4_b16x2_seed3407'

optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

