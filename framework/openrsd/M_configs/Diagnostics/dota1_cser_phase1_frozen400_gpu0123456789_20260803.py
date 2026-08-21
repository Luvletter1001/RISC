_base_ = (
    "../Step2_A10_Large_Pretrain_Stage3/"
    "A10_flex_rtm_v3_1_formal.py"
)

work_dir = (
    "/data1/zcy/OpenRSD/work_dirs/"
    "dota1_cser_phase1_frozen400_gpu0123456789_20260803"
)
load_from = (
    "/data1/zcy/OpenRSD_results/results/"
    "MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth"
)
resume = False
find_unused_parameters = False
randomness = dict(seed=2024, deterministic=False)

# The runner applies this allowlist before constructing the optimizer.
trainable_parameters = ["bbox_head.counter_support_ratio"]
frozen_parameters = None

runtime_root = "/data1/zcy/OpenRSD/data_local/cser_phase1_20260803"
dota2_support = (
    "/data1/zcy/datasets/OPENRSD_DATA/DOTAV2train/"
    "Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"
)
dota1_support = f"{runtime_root}/runtime_meta/dota1_visual_text_support.pkl"

dota1_classes = [
    "baseball-diamond",
    "basketball-court",
    "bridge",
    "ground-track-field",
    "harbor",
    "helicopter",
    "large-vehicle",
    "plane",
    "roundabout",
    "ship",
    "small-vehicle",
    "soccer-ball-field",
    "storage-tank",
    "swimming-pool",
    "tennis-court",
]
val_metainfo = dict(classes=dota1_classes, palette=[(220, 20, 60)])

custom_imports = dict(
    imports=[
        "M_AD.engine.runner.meta_remove_runer",
        "M_AD.engine.hooks.freeze_norm_stats_hook",
        "M_AD.datasets.transforms.formatting",
        "M_AD.datasets.transforms.loading",
        "M_AD.datasets.transforms.transforms",
        "M_AD.datasets.dota_online_v1",
        "M_AD.datasets.samplers.one_task_sampler",
        "M_AD.models.task_modules.assigners.safe_dynamic_soft_label_assigner",
        "M_AD.models.detectors.Flex_Rtmdet_v3_1_formal",
        "M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1",
        "M_AD.models.roi_heads.CLIP_VP_head_v1",
        "M_AD.models.necks.promopt_cspnext_pafpn",
        "M_AD.evaluation.metrics.detail_dota_metric",
    ],
    allow_failed_imports=False,
)

model = dict(
    support_feat_dict=dict(
        _delete_=True,
        Data1_DOTA2=dota2_support,
        Data1_DOTA1=dota1_support,
    ),
    pca_meta_pth=f"{runtime_root}/runtime_meta/pca_meta_unused.pkl",
    normalized_class_dict=(
        f"{runtime_root}/runtime_meta/normalized_class_dict.pkl"
    ),
    neg_support_data=(
        f"{runtime_root}/runtime_meta/neg_supports_unused.pkl"
    ),
    val_dataset_flag="Data1_DOTA1",
    val_support_classes=dota1_classes,
    support_type="visual",
    num_val_prompts=8,
    with_random_neg=False,
    with_image_rec_losses=False,
    with_aux_bbox_head=False,
    bbox_head=dict(
        counter_support_ratio=dict(
            enable=True,
            rank=8,
            init_strength=0.0,
        )
    ),
)

img_scale = (832, 832)
train_pipeline = [
    dict(type="mmdet.LoadImageFromFile", file_client_args=dict(backend="disk")),
    dict(type="LoadAnnotationsOnline", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxTypeSafe", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(type="mmdet.Resize", scale=img_scale, keep_ratio=True),
    dict(
        type="mmdet.RandomFlip",
        prob=0.75,
        direction=["horizontal", "vertical", "diagonal"],
    ),
    dict(type="RandomRotate", prob=0.5, angle_range=180),
    dict(
        type="mmdet.Pad",
        size=img_scale,
        pad_val=dict(img=(114, 114, 114)),
    ),
    dict(type="PackDetInputsMM"),
]
train_dataset = dict(
    type="DOTADatasetOnline",
    dataset_flag="Data1_DOTA2",
    data_root="/data/zcy/dataset/trainval_ms_full",
    metainfo=dict(classes=["SkyScript"], palette=[(220, 20, 60)]),
    data_prefix=dict(img_path="images/"),
    ann_file=f"{runtime_root}/train_formatted_pkls",
    embed_dims=256,
    img_shape=img_scale,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline,
)
train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type="OneTaskSampler",
        batch_size=2,
        source_prob=[1.0],
        num_gpus=10,
        max_iter_per_epoch=400,
    ),
    dataset=dict(type="mmdet.ConcatDataset", datasets=[train_dataset]),
)

val_img_scale = (800, 800)
val_pipeline = [
    dict(type="mmdet.LoadImageFromFile", file_client_args=dict(backend="disk")),
    dict(type="mmdet.Resize", scale=val_img_scale, keep_ratio=True),
    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(
        type="mmdet.Pad",
        size=val_img_scale,
        pad_val=dict(img=(114, 114, 114)),
    ),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]
val_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="DOTADataset",
        data_root=f"{runtime_root}/s2_val_view",
        metainfo=val_metainfo,
        ann_file="annfiles",
        data_prefix=dict(img_path="images"),
        img_shape=val_img_scale,
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(type="DETAILDOTAMetric", metric="mAP")
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1.0e-3, weight_decay=0.0),
)
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0,
        by_epoch=False,
        begin=0,
        end=1,
    )
]

custom_hooks = [dict(type="FreezeNormStatsHook", log_each_epoch=True)]
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=10),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=1),
)

