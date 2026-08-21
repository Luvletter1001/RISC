_base_ = './hrrsd_p15n_g_noencoder_topk1_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15o_decoder_light_hrrsd_20260624/'
    'p15o_b_dec2_noencoder_topk1_overfit100_4gpu_b8_v5')

# P15O-B is the more aggressive decoder-light variant. It keeps strict E2E
# inference and the P15N-G no-encoder/topk1 initializer, with only two decoder
# refinement layers.
model = dict(
    decoder=dict(num_layers=2))

randomness = dict(seed=3407, deterministic=False)
