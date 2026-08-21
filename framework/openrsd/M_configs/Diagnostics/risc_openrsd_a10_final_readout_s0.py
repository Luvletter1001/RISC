"""Interface-only S0 config for the frozen OpenRSD A10 architecture.

This config verifies construction and zero-alpha wiring. It inherits historical
dataset/support paths and is not an authorized N0-O or scientific evaluation
mouth. A separate immutable run manifest must provide the sealed filtered6605,
scale1024, text7 protocol before GPU inference.
"""

_base_ = [
    '../Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py'
]

model = dict(
    bbox_head=dict(
        risc_final_readout=dict(
            enabled=True,
            rank=8,
            init_alpha=0.0,
            max_alpha=0.1,
            max_delta_norm_ratio=0.05,
            init_seed=20260822,
        )))
