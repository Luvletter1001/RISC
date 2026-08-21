_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p8b_ap_projection_eval_gpu67.py'

custom_imports = dict(
    imports=[
        'M_AD.models.dense_heads.gs3c_rtmdet_head',
        'M_AD.models.backbones.gaussian_support_cspnext',
    ],
    allow_failed_imports=False)

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'eval_epoch3_plus2_bass_gsf_p9a_gst_backbone_head_full')

hrrsd_class_names = [
    'T', 'airplane', 'baseball', 'basketball', 'bridge', 'crossroad',
    'ground', 'harbor', 'parking', 'ship', 'storage', 'tennis', 'vehicle'
]
geometry_priors_csv = (
    'work_dirs/gs3c_dataset_inventory_20260619/'
    'hrrsd_internal_train_geometry_priors.csv')

model = dict(
    backbone=dict(
        type='GaussianSupportCSPNeXt',
        gaussian_support_adapter=dict(
            enable=True,
            class_geometry_priors_csv=geometry_priors_csv,
            class_names=hrrsd_class_names,
            num_classes=len(hrrsd_class_names),
            stages=(2, 3, 4),
            token_dim=32,
            temperature=0.75,
            gamma_init=0.03,
            residual_scale=0.5)),
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            level_routing=dict(
                enable=True,
                level_ref_sizes=[48.0, 128.0, 320.0],
                weight=0.12,
                max_bias_abs=0.25,
                target_assignment_source='pre_routing',
                learnable_gate_enable=True,
                learnable_gate_init=0.50,
                projection_loss_enable=False,
                projection_loss_weight=0.0,
                projection_loss_max_drop=0.01,
                projection_loss_min_assign_metric=0.0,
                log_stats=True))))
