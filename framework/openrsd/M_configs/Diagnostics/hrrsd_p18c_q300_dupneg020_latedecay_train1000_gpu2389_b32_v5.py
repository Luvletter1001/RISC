_base_ = './hrrsd_p16d_p15o_b_dec2_resnet50init_objneg_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p18_strict_e2e_scaleup_hrrsd_20260626/'
    'p18c_q300_dupneg020_latedecay_train1000_gpu2389_b32_v5')

# Backup for P18A/B: keep the stronger q300 set capacity from P16D and add
# P16T's duplicate hard-negative with late decay. This checks whether q200
# becomes too tight when moving from 500 to 1000 train images.
model = dict(
    bbox_head=dict(
        duplicate_rank_topk=0,
        duplicate_rank_loss_weight=0.0,
        matching_duplicate_neg_topk=2,
        matching_duplicate_neg_loss_weight=0.20,
        matching_duplicate_neg_warmup_iters=120,
        matching_duplicate_neg_decay_start_iters=440,
        matching_duplicate_neg_decay_end_iters=520,
        matching_duplicate_neg_decay_final_mult=0.5,
        matching_duplicate_neg_max_cost=6.0,
        cardinality_loss_weight=0.0))

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(indices=1000))

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        indices=1000,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/')))
test_dataloader = val_dataloader

auto_scale_lr = dict(enable=False, base_batch_size=128)

randomness = dict(seed=3407, deterministic=False)
