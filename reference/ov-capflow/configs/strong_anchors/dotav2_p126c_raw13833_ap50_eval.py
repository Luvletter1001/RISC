"""Replay the P126C strong DOTA-v2.0 anchor on raw 13,833-image val.

The model and dataset definition deliberately come from the exact historical
P126C experiment tree that produced the checkpoint.  This wrapper changes
only the output location and validation throughput settings.
"""

_base_ = (
    '/data1/zcy/GSOVD/.lab/workspace/'
    'p126c_p121_semantic_state_b12x4_80e_val20_fullval_gpu0189_20260708.py'
)

work_dir = (
    'work_dirs/dotav2_ap50_strong_anchor/'
    'p126c_epoch40_raw13833_gpu4567'
)

# Training-only hooks in the parent config write diagnostics to the historical
# GSOVD work directory.  They are unnecessary for a test-only replay.
custom_hooks = []

# The historical replay used batch size 2 per GPU and only ~1.1 GiB of logged
# evaluation memory.  Batch 32 keeps the four requested A40s busy while leaving
# substantial headroom.  Reducing this value is the only allowed OOM recovery.
test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
)
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
)
