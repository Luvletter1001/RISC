_base_ = './dota2_p24o_train1000_arbor_q300_memdn_gpu45_20260628.py'

work_dir = (
    'work_dirs/p24_arbor_dota2_train1000_gpu45_20260628/'
    'p24p_arbor_q300_memory_adapter_dnpos_decoder3_b16x2_seed3407')

# P24P tests whether the combined memory/query-prior model benefits from one
# extra rotated decoder refinement layer.
model = dict(
    decoder=dict(num_layers=3),
)

