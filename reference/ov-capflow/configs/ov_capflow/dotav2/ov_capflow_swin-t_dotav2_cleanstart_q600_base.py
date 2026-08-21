_base_ = ['./ov_capflow_swin-t_dotav2_c0_native_1e.py']

classes = (
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank',
    'swimming-pool', 'tennis-court')
novel_classes = ('airport', 'container-crane', 'helipad', 'helicopter')
base_classes = tuple(
    label for label in classes if label not in novel_classes)
metainfo = dict(classes=classes)
num_queries = 600
batch_size = 1

cleanstart_source = (
    '/data/zcy/GroundingDINO/weights/groundingdino_swint_ogc.pth')
cleanstart_source_sha256 = (
    '3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799')
cleanstart_provenance = (
    'work_dirs/dotav2_cleanstart/checkpoints/'
    'groundingdino_swint_ogc_q600_provenance.json')
load_from = (
    'work_dirs/dotav2_cleanstart/checkpoints/'
    'groundingdino_swint_ogc_q600_compatible.pth')
resume = False

prompt_protocol = dict(
    classification='text_token_similarity',
    fixed_classifier_width=False,
    separator='. ',
    canonical_classes=classes,
    base_classes=base_classes,
    novel_classes=novel_classes)

model = dict(
    type='OVCapFlow',
    num_queries=num_queries,
    train_query_groups=1,
    encoder=dict(num_cp=0),
    backbone=dict(init_cfg=None),
    bbox_head=dict(
        type='OVCapFlowHead',
        num_classes=len(classes),
        matching_query_groups=1),
    test_cfg=dict(_delete_=True))

train_dataloader = dict(
    batch_size=batch_size,
    batch_sampler=dict(
        type='DNQueryBudgetBatchSampler',
        num_matching_queries=num_queries,
        num_dn_queries=100,
        max_query_area=50000000,
        audit_path=(
            'work_dirs/dotav2_cleanstart/audits/'
            'full_train_epoch.json')),
    dataset=dict(
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False)))
val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=True))
test_dataloader = val_dataloader

train_cfg = dict(
    _delete_=True, type='VariableBatchEpochBasedTrainLoop', max_epochs=24,
    val_interval=6)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0 / 10, by_epoch=False,
        begin=0, end=500),
    dict(
        type='CosineAnnealingLR', by_epoch=True, begin=0, end=24,
        eta_min=1e-6),
]
optim_wrapper = dict(
    type='OptimWrapper',
    accumulative_counts=8,
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1),
    }))

randomness = dict(seed=20260715, deterministic=False, diff_rank_seed=False)
work_dir = 'work_dirs/dotav2_cleanstart/q600_base_protocol'
