_base_ = './dota2_p24a_train1000_arbor_replay_gpu45_20260628.py'

# P24E: increase the lightweight strict-E2E decoder from two to three layers.
# This spends modest extra compute on iterative box/angle refinement while
# preserving the Arbor optimizer and 1000-subset protocol.
work_dir = (
    'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/'
    'p24e_arbor_decoder3_b16x2_seed3407')

model = dict(
    decoder=dict(num_layers=3))
