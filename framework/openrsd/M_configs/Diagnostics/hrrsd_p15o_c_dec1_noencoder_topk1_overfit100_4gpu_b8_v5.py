_base_ = './hrrsd_p15n_g_noencoder_topk1_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15o_decoder_light_hrrsd_20260624/'
    'p15o_c_dec1_noencoder_topk1_overfit100_4gpu_b8_v5')

# P15O-C is the lower-bound speed probe: one decoder refinement layer after
# P15N-G's support-conditioned query initializer. If mAP collapses here but
# P15O-A/B survive, decoder depth is still carrying localization quality.
model = dict(
    decoder=dict(num_layers=1))

randomness = dict(seed=3407, deterministic=False)
