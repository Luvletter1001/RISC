from mmengine import Config


G3_V2_CFG = (
    "M_configs/Diagnostics/"
    "hrrsd_rtmdet_l_dota_init_internal_gs3c_g3_v2_apsensitive_train.py"
)
CTRL_CFG = (
    "M_configs/Diagnostics/"
    "hrrsd_rtmdet_l_dota_init_internal_gs3c_g3_v2_nog3_control_train.py"
)


def test_g3_v2_and_control_configs_are_matched_except_g3_gate():
    g3 = Config.fromfile(G3_V2_CFG)
    ctrl = Config.fromfile(CTRL_CFG)

    assert g3.randomness["seed"] == ctrl.randomness["seed"] == 3407
    assert g3.max_epochs == ctrl.max_epochs == 2
    assert g3.train_cfg["max_epochs"] == ctrl.train_cfg["max_epochs"] == 2
    assert g3.load_from == ctrl.load_from

    g3_scale = g3.model["bbox_head"]["gaussian_semantic_scale"]
    ctrl_scale = ctrl.model["bbox_head"]["gaussian_semantic_scale"]
    assert g3_scale["consistency_loss"]["enable"] is True
    assert g3_scale["consistency_loss"]["min_hardneg_logit"] == 0.0
    assert g3_scale["consistency_loss"]["max_gt_abs_z"] == 3.0
    assert ctrl_scale["enable"] is False
    assert ctrl_scale["consistency_loss"]["enable"] is False
