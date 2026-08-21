from experiments.rotation_semantic_attractor.src.metrics.rotation_gain import oracle_best_view_rows


def test_oracle_best_view_uses_actual_angle_rows():
    rows = [
        {"model_name": "m", "tile_id": "t", "angle": 0, "region_mode": "all_region", "false_sv_ratio": 0.6, "fr_sv": 0.7, "det_per_img": 10},
        {"model_name": "m", "tile_id": "t", "angle": 90, "region_mode": "all_region", "false_sv_ratio": 0.2, "fr_sv": 0.3, "det_per_img": 8},
    ]
    out = oracle_best_view_rows(rows)
    assert out[0]["oracle_best_angle"] == 90
    assert out[0]["oracle_gain_vs_angle000"] == 0.39999999999999997
    assert out[0]["uses_gt_or_angle_oracle"] is True
