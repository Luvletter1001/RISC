_base_ = './dota2_p24o_train1000_arbor_q300_memdn_gpu45_20260628.py'

work_dir = (
    'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/'
    'p24q_arbor_q300_memory_adapter_dnpos_hidden1024_b16x2_seed3407')

# P24Q widens the token-wise memory adapter to test whether the adapter is
# capacity-limited when paired with q300 and query positional priors.
model = dict(
    memory_adapter_hidden_channels=1024,
)

