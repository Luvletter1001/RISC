from mmengine import Config


def test_p4_ep2_clean_dump_config_keeps_gt_to_hardneg_pair_direction():
    cfg = Config.fromfile(
        "M_configs/Diagnostics/p4_ep2_clean_path_logits_s3c_gpu45.py")
    probe = cfg.model.bbox_head.scale_semantic_calibration.ep2_path_probe

    assert probe.enable is True
    assert probe.output_csv.endswith("ep2_pre_nms_raw_logits.csv")
    assert ("tennis-court", "small-vehicle") in probe.class_pairs
    assert ("small-vehicle", "tennis-court") not in probe.class_pairs
    assert probe.max_locations_per_level == 128


def test_p4_ep2_env_dump_config_accepts_output_override(monkeypatch):
    monkeypatch.setenv(
        "EP2_PATH_PROBE_CSV",
        "work_dirs/custom_ep2/ep2_pre_nms_raw_logits.csv")
    cfg = Config.fromfile(
        "M_configs/Diagnostics/p4_ep2_clean_path_logits_s3c_env.py")
    probe = cfg.model.bbox_head.scale_semantic_calibration.ep2_path_probe

    assert probe.output_csv == (
        "work_dirs/custom_ep2/ep2_pre_nms_raw_logits.csv")


def test_p4_ep2_env_dump_config_accepts_max_locations_override(monkeypatch):
    monkeypatch.setenv("EP2_PATH_PROBE_MAX_LOCATIONS", "2048")

    cfg = Config.fromfile(
        "M_configs/Diagnostics/p4_ep2_clean_path_logits_s3c_env.py")
    probe = cfg.model.bbox_head.scale_semantic_calibration.ep2_path_probe

    assert probe.max_locations_per_level == 2048
