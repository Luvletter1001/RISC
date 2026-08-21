_base_ = './dota2_p23c_strict_e2e_full5e_from_p22b_b4x2_v1.py'

# P23D: same full-DOTA2 strict-E2E recovery setup as P23C, after replacing
# the P15 2x2 Gaussian determinant with an explicit analytic formula to avoid
# the CUDA MAGMA queue assertion hit by torch.linalg.det in P23C.
work_dir = (
    'work_dirs/p23_strict_e2e_dota2_full_20260626/'
    'p23d_full5e_from_p22b_b4x2_analyticgwd_v1')
