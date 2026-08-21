_base_ = './hrrsd_p15n_g_noencoder_topk1_overfit100_4gpu_b8_v5.py'

work_dir = (
    'work_dirs/p15o_decoder_light_hrrsd_20260624/'
    'p15o_a_dec3_noencoder_topk1_overfit100_4gpu_b8_v5')

# P15O-A keeps the P15N-G strict E2E path unchanged, but cuts the DINO-style
# decoder from 6 layers to 3 layers. DINO injects the matching head branch count
# at build time: num_pred_layer = decoder + 1.
model = dict(
    decoder=dict(num_layers=3))

randomness = dict(seed=3407, deterministic=False)
