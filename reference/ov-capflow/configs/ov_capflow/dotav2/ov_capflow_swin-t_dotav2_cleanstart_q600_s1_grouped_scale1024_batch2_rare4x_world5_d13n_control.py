_base_ = [
    './ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x.py'
]

physical_gpus = (5, 6, 7, 8, 9)
selected_world_size = 5
d13n_pair_id = 'D13-N-Q600-existence-residual-world5-seed20260716'
d13n_role = 'control'
d13n_mouth = 'proxy400'
d13n_test_only = False
d13n_master_port = 29842
d13n_updates_per_epoch = 160
d13n_total_updates = 1920
d13n_parent_sha256 = (
    'a4f2661e6c1645b08f296dfb2bbebfd76afe8c366d6152dbbc333bbf840af6b8')
d13n_train_manifest_sha256 = (
    '1457c641d91a7e0bf26a62f0cd6c70c71d9e9c6df5a2137fe4b73b8e8fc05290')
d13n_proxy_manifest_sha256 = (
    'a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e')
d13n_audit_path = (
    'work_dirs/dotav2_cleanstart/audits/'
    'd13n_world5_control_seed20260716_gpu56789_epoch_{epoch:02d}.json')

model = dict(
    decoder=dict(
        enable_null_reservoir=False,
        layer_cfg=dict(
            enable_semantic_fusion=False,
            enable_density_capacity=False)),
    bbox_head=dict(
        matching_query_groups=3,
        balanced_cfg=dict(enabled=False),
        position_supervised_cfg=dict(enabled=False),
        adaptive_dn_cfg=dict(enabled=False),
        readout_cfg=dict(
            temperature=1.0,
            power=1.0,
            use_capacity=False),
        existence_loss_weight=0.0),
    density_loss_cfg=dict(weight=0.0),
    null_loss_cfg=dict(),
    freeze_except_patterns=[
        r'^bbox_head\.existence_residual\.(weight|bias)$'
    ])

train_dataloader = dict(
    batch_size=2,
    batch_sampler=dict(
        update_count_multiple=1,
        audit_path=d13n_audit_path,
        audit_noreplace=True))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    constructor='D13NOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=1)

default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=-1,
        save_best=None,
        rule=None,
        save_last=True))
custom_hooks = [dict(type='D13NParentEvalModeHook', role='control')]

randomness = dict(
    seed=20260716,
    deterministic=False,
    diff_rank_seed=False)
load_from = (
    'work_dirs/dotav2_cleanstart/'
    'full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/'
    'epoch_24.pth')
work_dir = (
    'work_dirs/dotav2_cleanstart/'
    'd13n_world5_control_seed20260716_gpu56789_batch2')
resume = False
