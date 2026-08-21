_base_ = (
    '../../work_dirs/focus_ovd_a10_mess_fourier_dual_text_gpu67_mid_gate_best_bs1_20260612/'
    'best_eval/focus_ovd_a10_mess_fourier_dual_text_gpu67_mid_gate_best_bs1_20260612.py')

dota2_classes = [
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank', 'swimming-pool',
    'tennis-court',
]

data_root = '/data1/zcy/datasets/DOTA2_1024_500/'
support_root = (
    '/data1/zcy/OpenRSD/work_dirs/'
    'dotav2_p4_lowtext_lser_sise_gpu67_20260617_153546/eval_bundle')
val_img_scale = (800, 800)
file_client_args = dict(backend='disk')

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=val_img_scale, keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad',
        size=val_img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

work_dir = (
    'work_dirs/p21_strict_e2e_dota2_subset_20260626/'
    'baseline_openrsd_eval_train500_gpu45_v1')

load_from = (
    '/data1/zcy/OpenRSD/work_dirs/'
    'focus_ovd_a10_mess_fourier_dual_text_gpu67_mid_gate_best_bs1_20260612/'
    'best_dota_mAP_epoch_12.pth')

model = dict(
    neg_support_data=f'{support_root}/Neg_supports_v2.pkl',
    normalized_class_dict=f'{support_root}/normalized_class_dict.pkl',
    pca_meta_pth=f'{support_root}/7_25_pca_meta_DINOv2_256.pkl',
    support_feat_dict=dict(
        Data1_DOTA2=(
            f'{support_root}/DOTA2_1024_500/ss_train/'
            'Step5_3_Prepare_Visual_Text_DINOv2_support.pkl')),
    val_dataset_flag='Data1_DOTA2',
    val_support_classes=dota2_classes)

val_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='ss_train/annfiles',
        data_prefix=dict(img_path='ss_train/images'),
        metainfo=dict(classes=dota2_classes, palette=[(220, 20, 60)]),
        img_shape=val_img_scale,
        indices=500,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = val_dataloader
val_evaluator = dict(type='DETAILDOTAMetric', metric='mAP')
test_evaluator = val_evaluator
