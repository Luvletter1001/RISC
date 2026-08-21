_base_ = './hrrsd_rtmdet_l_dota_init_internal_bass_gsf_p5_posdelta_train_gpu67.py'

geometry_priors_csv = (
    'work_dirs/gs3c_dataset_inventory_20260619/'
    'hrrsd_internal_train_geometry_priors.csv')

work_dir = (
    'work_dirs/gs3c_hrrsd_rtmdetl_dota_init_20260619/'
    'train_epoch3_plus2_bass_gsf_p5b_geom')

model = dict(
    bbox_head=dict(
        gaussian_semantic_scale=dict(
            geometry_support_enable=True,
            class_geometry_priors_csv=geometry_priors_csv,
            density_head=dict(
                use_geometry_logprob=True,
                delta_loss_min_hardneg_score=0.10))))
