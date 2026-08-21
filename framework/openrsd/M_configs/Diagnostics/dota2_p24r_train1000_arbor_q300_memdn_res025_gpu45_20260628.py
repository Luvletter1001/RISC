_base_ = './dota2_p24o_train1000_arbor_q300_memdn_gpu45_20260628.py'

work_dir = (
    'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/'
    'p24r_arbor_q300_memory_adapter_dnpos_res025_b16x2_seed3407')

# P24R weakens the memory-adapter residual to test whether the combined model
# over-corrects the encoder-bypass memory when the query prior is also active.
model = dict(
    memory_adapter_residual_scale=0.25,
)

