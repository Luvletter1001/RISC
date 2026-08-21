_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p6_supportneg_eval_gpu67.py'

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'eval_epoch3_plus2_bass_gsf_p7_scalecons_head_full')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            adapter_nonpositive_delta=False,
            geometry_support_enable=True,
            class_geometry_priors_csv=(
                'work_dirs/gs3c_dataset_inventory_20260619/'
                'hrrsd_internal_train_geometry_priors.csv'),
            density_head=dict(
                enable=True,
                max_delta_abs=0.25,
                nonpositive_delta=False,
                apply_logit_delta=False,
                logit_delta_source='head',
                use_geometry_logprob=True,
                delta_loss_enable=False,
                positive_delta_loss_enable=False,
                pair_margin_loss_enable=False,
                support_negative_loss_enable=False,
                scale_consistency_loss_enable=False))))
